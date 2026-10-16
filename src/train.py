import contextlib
import hashlib
import json
import math
import signal
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, BatchSampler
from tqdm.auto import tqdm
from .checkpoint import (
    load_adapted,
    save_training,
    select_resume_path,
    CosineSchedule,
    restore_schedule,
    atomic_copy,
)
from .data import Windows
from .evaluate import validation_fgd
from .generate import GenerationStopped
from .losses import flow_loss
from .model import MotionModel
from .runtime import (
    seed_all,
    batch_to,
    autocast_context,
    grad_scaler,
    write_json,
    barrier,
    rng_state,
    restore_rng,
    sampler_identity,
    validation_identity,
)


class ResumeBatchSampler:
    def __init__(self, sampler, batch_size):
        self.batches = BatchSampler(sampler, batch_size, drop_last=False)
        self.start_batch = 0

    def __iter__(self):
        for i, indices in enumerate(self.batches):
            if i >= self.start_batch:
                yield indices

    def __len__(self):
        return max(0, len(self.batches) - self.start_batch)


def restore_training(state, optimizer, scaler, cfg, prepared, progress, device, rank, world, loader_rng, order_hash):
    if state["world_size"] != world:
        raise ValueError("Full resume requires the saved GPU count")
    keys = (
        "BATCH_SIZE_PER_GPU",
        "GRADIENT_ACCUMULATION",
        "TRAIN_STRIDE_FRAMES",
        "USE_ADDITIONAL",
        "SEED",
        "MIXED_PRECISION",
        "FP16_GRAD_SCALER",
        "CFG_DROPOUT",
        "GEOMETRIC_LOSS_WEIGHT",
        "STRUCTURE_LOSS_WEIGHT",
        "EXPRESSION_LOSS_WEIGHT",
        "TEMPORAL_CONTRASTIVE_WEIGHT",
        "MAX_GRAD_NORM",
    )
    for key in keys:
        if state["config"][key] != cfg[key]:
            raise ValueError(
                f"Full resume requires unchanged {key}; use INIT_ADAPTED_PATH to start a new experiment"
            )
    if state["config"].get("GPU_RESIDENT", False):
        raise ValueError("Full resume cannot switch from GPU-resident sampling to CPU streaming; use INIT_ADAPTED_PATH with a new output folder. Evaluation works directly with the checkpoint.")
    if state["prepared_signature"] != Path(prepared).name and not state.get("training_order_sha256"):
        raise ValueError("Cannot verify data order for this checkpoint")
    if state.get("training_order_sha256", order_hash) != order_hash:
        raise ValueError("Training window order changed")
    if state["progress"]["selected_fgd_sampler"] != sampler_identity(cfg):
        raise ValueError("Resume sampler changed")
    if state["progress"].get(
        "validation_selection", validation_identity(state["config"])
    ) != validation_identity(cfg):
        raise ValueError("Validation selection changed")
    fused = optimizer.param_groups[0]["fused"]
    optimizer.load_state_dict(state["optimizer"])
    if scaler.is_enabled():
        if not state.get("scaler"):
            raise ValueError("FP16 full resume requires saved GradScaler state")
        scaler.load_state_dict(state["scaler"])
    for group in optimizer.param_groups:
        group["fused"] = fused
        group["foreach"] = False if fused else None
    scheduler = restore_schedule(optimizer, state, cfg)
    progress.update(state["progress"])
    restore_rng(state["rng_by_rank"][rank], device)
    if state.get("loader_rng_by_rank"):
        loader_rng.set_state(state["loader_rng_by_rank"][rank])
    return scheduler


def validate_epoch(model, loader, cfg, prepared, device, epoch, progress, stop):
    saved_rng = rng_state(device)
    try:
        model.eval()
        seed_all(cfg["SEED"] + 9000)
        total = count = 0
        with torch.inference_mode():
            for batch in loader:
                if stop():
                    raise GenerationStopped("Validation stopped")
                with autocast_context(cfg, device):
                    terms = flow_loss(model, batch_to(batch, device), cfg, False)
                total += float(terms["total"]) * len(batch["x"])
                count += len(batch["x"])
        report = None
        if (epoch + 1) % cfg["VAL_FGD_EVERY_EPOCHS"] == 0 or epoch + 1 == cfg["EPOCHS"]:
            report = validation_fgd(model, cfg, prepared, device, should_stop=stop)
            report.update(epoch=epoch, completed_epochs=epoch + 1, update=progress["updates"])
            write_json(Path(cfg["OUTPUT_DIR"]) / "validation_latest.json", report)
        return {"val_loss": total / count, "val_fgd": report["FGD_raw"] if report else None}
    except GenerationStopped:
        return {"stop": stop() or "user_stop"}
    finally:
        restore_rng(saved_rng, device)
        model.train()


def train(cfg, prepared, rank, world, device):
    out = Path(cfg["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP_TRAINING"
    seed_all(cfg["SEED"] + rank)
    resume_path = select_resume_path(cfg)
    if not resume_path and (out / "last.pt").exists():
        raise ValueError("Use AUTO_RESUME or a fresh output folder")
    state = None
    if resume_path:
        model, state = load_adapted(cfg, resume_path, device, prepared)
    elif cfg["INIT_ADAPTED_PATH"]:
        model, _ = load_adapted(cfg, cfg["INIT_ADAPTED_PATH"], device, prepared, weights_only=True)
    else:
        model = MotionModel(cfg).to(device)
    wrapped = (
        DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else model
    )
    fused = cfg["USE_FUSED_ADAM"] and device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"], fused=fused)
    scaler = grad_scaler(cfg, device)
    progress = {
        "epoch": 0,
        "next_batch": 0,
        "updates": 0,
        "skipped_steps": 0,
        "best_val_fgd": None,
        "best_val_loss": None,
        "train_seconds": 0.0,
        "selected_fgd_sampler": sampler_identity(cfg),
        "validation_selection": validation_identity(cfg),
        "last_validation_update": None,
    }
    dataset = Windows(cfg, prepared, ["train"] + (["additional"] if cfg["USE_ADDITIONAL"] else []))
    order_hash = hashlib.sha256(json.dumps(dataset.rows, separators=(",", ":")).encode()).hexdigest()
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg["SEED"], drop_last=False)
    batches = ResumeBatchSampler(sampler, cfg["BATCH_SIZE_PER_GPU"])
    total_batches = len(batches.batches)
    loader_rng = torch.Generator().manual_seed(cfg["SEED"] + 2000 + rank)
    options = {"prefetch_factor": 2} if cfg["NUM_WORKERS"] else {}
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["NUM_WORKERS"] > 0,
        generator=loader_rng,
        **options,
    )
    updates = math.ceil(total_batches / cfg["GRADIENT_ACCUMULATION"])
    schedule_length = cfg["COSINE_T_MAX_UPDATES"] or cfg["MAX_UPDATES"] or cfg["EPOCHS"] * updates
    scheduler = (
        CosineSchedule(optimizer, schedule_length, cfg["COSINE_MIN_LR"])
        if cfg["LR_SCHEDULER"] == "cosine"
        else None
    )
    val_loader = None
    if rank == 0 and cfg["RUN_VALIDATION"]:
        valset = Windows(cfg, prepared, ["val"])
        val_rng = torch.Generator().manual_seed(cfg["SEED"] + 9000)
        indices = torch.randperm(len(valset), generator=val_rng)[
            : cfg["VAL_LOSS_MAX_BATCHES"] * cfg["BATCH_SIZE_PER_GPU"]
        ].tolist()
        val_loader = DataLoader(
            Subset(valset, indices), batch_size=cfg["BATCH_SIZE_PER_GPU"], num_workers=0, generator=val_rng
        )
    if state is not None:
        scheduler = restore_training(
            state, optimizer, scaler, cfg, prepared, progress, device, rank, world, loader_rng, order_hash
        )
        if rank == 0:
            for name in ("best_val_fgd.pt", "best_val_loss.pt"):
                old = Path(resume_path).parent / name
                if old.is_file() and old.resolve() != (out / name).resolve():
                    atomic_copy(old, out / name)
            print(
                f"Resumed epoch={progress['epoch']} batch={progress['next_batch']} updates={progress['updates']}",
                flush=True,
            )
        del state
    started = time.perf_counter()
    previous_seconds = progress["train_seconds"]
    progress["stop_reason"] = None
    optimizer.zero_grad(set_to_none=True)

    def save(reason):
        progress["train_seconds"] = previous_seconds + time.perf_counter() - started
        progress["save_reason"] = reason
        save_training(
            out / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            prepared,
            progress,
            rank,
            world,
            device,
            loader_rng,
            order_hash,
        )

    def stop():
        if stop_file.exists():
            return "user_stop"
        if (
            cfg["MAX_TRAIN_HOURS"] is not None
            and time.perf_counter() - started >= cfg["MAX_TRAIN_HOURS"] * 3600
        ):
            return "session_time_limit"
        if cfg["MAX_UPDATES"] is not None and progress["updates"] >= cfg["MAX_UPDATES"]:
            return "update_limit"
        return None

    def shared_stop():
        value = [stop() if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(value, src=0)
        return value[0]

    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        stop_file.touch()
        if rank == 0:
            print("Stopping after the current optimizer update and saving last.pt", flush=True)

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        reason = shared_stop()
        for epoch in range(progress["epoch"], cfg["EPOCHS"]):
            if reason:
                break
            sampler.set_epoch(epoch)
            batches.start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
            if not 0 <= batches.start_batch <= total_batches:
                raise ValueError("Invalid checkpoint batch cursor")
            model.train()
            totals = {}
            count = 0
            with tqdm(
                total=total_batches,
                initial=batches.start_batch,
                desc=f"Epoch {epoch + 1}/{cfg['EPOCHS']}",
                disable=rank != 0,
                dynamic_ncols=True,
            ) as bar:
                iterator = iter(loader)
                for index, batch in enumerate(iterator, start=batches.start_batch):
                    batch = batch_to(batch, device)
                    accumulation = cfg["GRADIENT_ACCUMULATION"]
                    group_size = min(accumulation, total_batches - index // accumulation * accumulation)
                    sync = (index + 1) % accumulation == 0 or index + 1 == total_batches
                    context = wrapped.no_sync() if world > 1 and not sync else contextlib.nullcontext()
                    with context:
                        with autocast_context(cfg, device):
                            terms = flow_loss(wrapped, batch, cfg, True)
                            loss = terms["total"] / group_size
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite loss")
                        scaler.scale(loss).backward()
                    if rank == 0:
                        for key, value in terms.items():
                            totals[key] = totals.get(key, 0.0) + float(value.detach())
                    count += 1
                    bar.update(1)
                    if not sync:
                        continue
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["MAX_GRAD_NORM"], error_if_nonfinite=False
                    )
                    invalid = torch.tensor(int(not torch.isfinite(norm)), device=device)
                    if world > 1:
                        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                    if invalid.item() and not scaler.is_enabled():
                        raise FloatingPointError("Nonfinite gradients")
                    if scaler.is_enabled():
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        skipped = scaler.get_scale() < previous_scale
                    else:
                        optimizer.step()
                        skipped = False
                    optimizer.zero_grad(set_to_none=True)
                    if not skipped:
                        if scheduler is not None:
                            scheduler.step()
                        progress["updates"] += 1
                    else:
                        progress["skipped_steps"] = progress.get("skipped_steps", 0) + 1
                    progress.update(epoch=epoch, next_batch=index + 1)
                    if rank == 0:
                        bar.set_postfix(
                            loss=f"{float(terms['total'].detach()):.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                            updates=progress["updates"],
                            scale=f"{scaler.get_scale():.0f}" if scaler.is_enabled() else "off",
                        )
                        if progress["updates"] % cfg["LOG_EVERY_UPDATES"] == 0:
                            row = {key: value / count for key, value in totals.items()}
                            row.update(
                                epoch=epoch,
                                update=progress["updates"],
                                lr=optimizer.param_groups[0]["lr"],
                                grad_norm=float(norm),
                                skipped_steps=progress.get("skipped_steps", 0),
                                grad_scale=scaler.get_scale() if scaler.is_enabled() else None,
                            )
                            with (out / "training.jsonl").open("a") as handle:
                                handle.write(json.dumps(row) + "\n")
                            totals, count = {}, 0
                    reason = shared_stop()
                    every = cfg["SAVE_EVERY_UPDATES"]
                    if reason or (not skipped and every and progress["updates"] % every == 0):
                        save(reason or "periodic_latest")
                    if reason:
                        break
            if reason:
                break
            best_loss = best_fgd = False
            if cfg["RUN_VALIDATION"]:
                save("before_validation")
                result = [
                    validate_epoch(model, val_loader, cfg, prepared, device, epoch, progress, stop)
                    if rank == 0
                    else None
                ]
                if world > 1:
                    dist.broadcast_object_list(result, src=0)
                values = result[0]
                if "stop" in values:
                    reason = values["stop"]
                    break
                best_loss = (
                    progress["best_val_loss"] is None or values["val_loss"] < progress["best_val_loss"]
                )
                best_fgd = values["val_fgd"] is not None and (
                    progress["best_val_fgd"] is None or values["val_fgd"] < progress["best_val_fgd"]
                )
                if best_loss:
                    progress["best_val_loss"] = values["val_loss"]
                if best_fgd:
                    progress["best_val_fgd"] = values["val_fgd"]
                    progress["best_val_fgd_update"] = progress["updates"]
                if values["val_fgd"] is not None:
                    progress["last_validation_update"] = progress["updates"]
            progress.update(epoch=epoch + 1, next_batch=0)
            save("epoch_complete")
            if rank == 0:
                if best_fgd:
                    atomic_copy(out / "last.pt", out / "best_val_fgd.pt")
                if best_loss and cfg["SAVE_BEST_VAL_LOSS"]:
                    atomic_copy(out / "last.pt", out / "best_val_loss.pt")
            barrier()
            reason = shared_stop()
        progress["stop_reason"] = reason or "epochs_complete"
        save("training_exit")
        if rank == 0:
            write_json(out / "training_complete.json", progress)
            print(f"Saved {out / 'last.pt'}", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
