import os
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .config import ARCHITECTURE_KEYS, DEFAULTS
from .model import MotionModel
from .losses import temporal_loss_state
from .runtime import stats_from, rng_state, barrier

FORMAT = "voxmove-motion-v1"


def select_resume_path(cfg):
    latest = Path(cfg["OUTPUT_DIR"]) / "last.pt"
    if cfg["AUTO_RESUME"] and latest.is_file():
        return str(latest)
    return cfg["RESUME_PATH"]


def read_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def resolve_checkpoint(path, output_dir):
    path = Path(path)
    if path.is_dir():
        path = path / "last.pt"
    if path.suffix.lower() == ".zip":
        dest = Path(tempfile.mkdtemp(prefix="resume_", dir=Path(output_dir)))
        with zipfile.ZipFile(path) as archive:
            if "last.pt" not in archive.namelist():
                raise ValueError("Resume bundle has no top-level last.pt")
            for name in ["last.pt", "best_val_fgd.pt", "best_val_loss.pt"]:
                if name in archive.namelist():
                    with archive.open(name) as src, (dest / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        path = dest / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_copy(source, target):
    target = Path(target)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


class CosineSchedule:
    def __init__(self, optimizer, total, minimum):
        self.optimizer = optimizer
        self.T_max = int(total)
        self.eta_min = float(minimum)
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.last_epoch = 0

    def step(self):
        self.last_epoch = min(self.last_epoch + 1, self.T_max)
        ratio = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.eta_min + (base - self.eta_min) * ratio

    def state_dict(self):
        return {
            "kind": "clamped_cosine",
            "T_max": self.T_max,
            "eta_min": self.eta_min,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
            "_last_lr": [float(g["lr"]) for g in self.optimizer.param_groups],
        }

    def load_state_dict(self, state):
        self.T_max = int(state["T_max"])
        self.eta_min = float(state["eta_min"])
        self.base_lrs = [float(x) for x in state["base_lrs"]]
        self.last_epoch = int(state["last_epoch"])
        if "_last_lr" in state:
            for group, lr in zip(self.optimizer.param_groups, state["_last_lr"]):
                group["lr"] = float(lr)


def restore_schedule(optimizer, state, cfg):
    saved = state.get("scheduler")
    if saved is not None:
        scheduler = CosineSchedule(optimizer, saved["T_max"], saved["eta_min"])
        scheduler.load_state_dict(saved)
        return scheduler
    if state.get("checkpoint_schema", 1) < 2:
        print(
            "[RESUME] Legacy checkpoint has no scheduler state; retaining its saved constant LR. Use INIT_ADAPTED_PATH to start a new schedule.",
            flush=True,
        )
    return None


def load_adapted(cfg, path, device, prepared=None, weights_only=False):
    state = read_checkpoint(path)
    full = isinstance(state, dict) and all((k in state for k in ("model", "config", "stats", "progress")))
    if not full and (not weights_only):
        raise ValueError(
            "Full resume/evaluation requires a complete checkpoint; use INIT_ADAPTED_PATH for bare weights"
        )
    if full:
        mismatch = {
            k: (cfg[k], state["config"].get(k)) for k in ARCHITECTURE_KEYS if cfg[k] != state["config"].get(k)
        }
        if mismatch:
            raise ValueError(f"Checkpoint architecture differs: {mismatch}")
        if prepared is not None:
            for key, value in stats_from(prepared).items():
                if not np.allclose(value, state["stats"][key], rtol=1e-06, atol=1e-07):
                    raise ValueError("Expression statistics differ from the checkpoint")
        weights = state["model"]
    else:
        weights = state.get("state_dict", state.get("model_state_dict", state.get("model", state)))
    weights = {k.removeprefix("module."): v for k, v in weights.items() if isinstance(v, torch.Tensor)}
    model = MotionModel(cfg).to(device)
    model.load_state_dict(weights, strict=True)
    if full:
        model._temporal_loss_state = state.get("temporal_loss_state")
    return (model, state if full and (not weights_only) else {})


def save_training(
    path, model, optimizer, scheduler, scaler, cfg, prepared, progress, rank, world, device, loader_rng, order_hash
):
    local = {"rng": rng_state(device), "loader": loader_rng.get_state()}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    if rank == 0:
        state = {
            "format": FORMAT,
            "checkpoint_schema": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "config": {k: cfg[k] for k in DEFAULTS},
            "stats": stats_from(prepared),
            "temporal_loss_state": temporal_loss_state(model),
            "progress": progress.copy(),
            "prepared_signature": Path(prepared).name,
            "training_order_sha256": order_hash,
            "world_size": world,
            "rng_by_rank": [s["rng"] for s in gathered],
            "loader_rng_by_rank": [s["loader"] for s in gathered],
        }
        atomic_torch_save(state, path)
    barrier()


def checkpoint_for_eval(cfg):
    path = (
        Path(cfg["CHECKPOINT_PATH"])
        if cfg["CHECKPOINT_PATH"]
        else Path(cfg["OUTPUT_DIR"]) / (cfg["FULL_TEST_CHECKPOINT"] + ".pt")
    )
    return resolve_checkpoint(path, cfg["OUTPUT_DIR"])
