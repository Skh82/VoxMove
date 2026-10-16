import os
import json
import random
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_context(cfg, device):
    mode = cfg["MIXED_PRECISION"]
    enabled = device.type == "cuda" and mode != "fp32"
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    if enabled and mode == "bf16" and (not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 unsupported on this GPU; choose fp16 or fp32")
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def grad_scaler(cfg, device):
    kwargs = {**cfg["FP16_GRAD_SCALER"], "enabled": device.type == "cuda" and cfg["MIXED_PRECISION"] == "fp16"}
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", **kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


def distributed_setup(cfg, command):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    evaluation = command == "evaluate"
    torch.set_num_threads(
        cfg["EVAL_CPU_THREADS"] if evaluation or command == "validate" else cfg["TRAIN_CPU_THREADS"]
    )
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        gpu_count = min(cfg["NUM_GPUS"], torch.cuda.device_count())
        gpu = local % gpu_count if evaluation else local
        if gpu >= gpu_count:
            raise ValueError("Training/preflight require one worker per GPU")
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" and (not evaluation) else "gloo", timeout=timedelta(hours=8)
        )
    if evaluation:
        print(f"Evaluation worker {rank}/{world}: {device}; CUDA graphs={cfg['USE_CUDA_GRAPHS']}", flush=True)
    return (rank, world, device)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def stats_from(prepared):
    with np.load(Path(prepared) / "stats.npz") as f:
        return {k: f[k] for k in ["expr_mean", "expr_std"]}


def validation_identity(cfg):
    return {k: cfg[k] for k in ["VAL_FGD_NUM_UTTERANCES", "VAL_FGD_SUBSET_SEED"]}


def sampler_identity(cfg):
    return {k: cfg[k] for k in ["WINDOW_FRAMES", "HISTORY_FRAMES", "FLOW_STEPS", "CFG_SCALE", "EVAL_SEED"]}


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if device.type == "cuda" and value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)
