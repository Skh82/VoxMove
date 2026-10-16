import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="VoxMove motion training and generation")
    parser.add_argument("command", choices=["all", "prepare", "train", "generate", "validate", "evaluate"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--speaker-id", type=int)
    args = parser.parse_args()
    from src.config import load_config

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["CHECKPOINT_PATH"] = str(Path(args.checkpoint).resolve())
    threads = (
        cfg["EVAL_CPU_THREADS"]
        if args.command in ("generate", "validate", "evaluate")
        else cfg["TRAIN_CPU_THREADS"]
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    import numpy as np
    import torch.distributed as dist
    from src.runtime import distributed_setup, barrier, write_json
    from src.checkpoint import select_resume_path, resolve_checkpoint, checkpoint_for_eval, load_adapted
    from src.data import prepare_data, data_signature

    rank, world, device = distributed_setup(cfg, args.command)
    out = Path(cfg["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "generate":
            if world != 1 or not args.input or not args.output:
                parser.error("generate requires one process, --input and --output")
            from src.generate import generate

            with np.load(args.input, allow_pickle=False) as data:
                if "hidden" not in data:
                    parser.error("Input NPZ must contain hidden [T, TT_DIM]")
                pid = (
                    args.speaker_id
                    if args.speaker_id is not None
                    else int(data["pid"])
                    if "pid" in data
                    else 0
                )
                sample = {
                    "hidden": np.asarray(data["hidden"], np.float32),
                    "pid": pid,
                    "uid": str(data["uid"].item()) if "uid" in data else Path(args.input).stem,
                    "seams": np.asarray(data["seams"], np.int64)
                    if "seams" in data
                    else np.empty(0, np.int64),
                }
            if (
                sample["hidden"].ndim != 2
                or sample["hidden"].shape[1] != cfg["TT_DIM"]
                or not len(sample["hidden"])
                or not np.isfinite(sample["hidden"]).all()
            ):
                raise ValueError("Invalid hidden feature shape or values")
            if not 0 <= pid < cfg["NUM_SPEAKERS"]:
                raise ValueError("Speaker ID is outside the model vocabulary")
            model, state = load_adapted(cfg, checkpoint_for_eval(cfg), device)
            stats = state["stats"]
            del state
            result = generate(model, sample, stats, cfg, device)
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **{k: v for k, v in result.items() if k != "inference"})
            print(f"Saved {target}", flush=True)
            return
        if args.command in ("all", "prepare"):
            if rank == 0:
                prepared = prepare_data(cfg)
                write_json(out / "prepared_paths.json", {"prepared": str(prepared)})
            barrier()
            if args.command == "prepare":
                return
        receipt = out / "prepared_paths.json"
        if not receipt.exists():
            raise FileNotFoundError("Run prepare or all first")
        prepared = Path(json.loads(receipt.read_text())["prepared"])
        if prepared.name != data_signature(cfg):
            raise ValueError("Data configuration changed; run prepare again")
        if args.command in ("train", "all"):
            cfg["RESUME_PATH"] = select_resume_path(cfg)
            resolved = [None]
            if rank == 0:
                resolved[0] = {
                    k: str(resolve_checkpoint(cfg[k], out))
                    for k in ("RESUME_PATH", "INIT_ADAPTED_PATH")
                    if cfg[k]
                }
                (out / "STOP_TRAINING").unlink(missing_ok=True)
                write_json(out / "run_config.json", cfg)
            if world > 1:
                dist.broadcast_object_list(resolved, src=0)
            cfg.update(resolved[0])
            barrier()
            from src.train import train

            train(cfg, prepared, rank, world, device)
        elif args.command == "validate":
            if world != 1:
                raise ValueError("Validation uses one process")
            from src.evaluate import validation_fgd

            model, state = load_adapted(cfg, checkpoint_for_eval(cfg), device, prepared)
            del state
            result = validation_fgd(model, cfg, prepared, device)
            write_json(out / "validation_manual.json", result)
            print(json.dumps(result, indent=2))
        elif args.command == "evaluate":
            from src.evaluate import evaluate

            selected = [str(checkpoint_for_eval(cfg)) if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(selected, src=0)
            evaluate(cfg, prepared, rank, world, device, selected[0])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
