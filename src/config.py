import json
import math
from pathlib import Path

DEFAULTS = {
    # Model, motion representation, and evaluation parameters.
    "FFN_VARIANT": "swiglu_parameter_matched",
    "SWIGLU_HIDDEN_DIM": 512,
    "USE_TT_HIDDEN": True,
    "USE_VOXTREAM_AUDIO_TOKENS": False,
    "USE_VOXTREAM_TEXT_TOKENS": False,
    "AUDIO_VOCAB": 2048,
    "PHONE_VOCAB": 512,
    "MODEL_WIDTH": 384,
    "MODEL_LAYERS": 6,
    "WINDOW_FRAMES": 50,
    "HISTORY_FRAMES": 16,
    "TRAIN_STRIDE_FRAMES": 16,
    "LONGFORM_MIN_SECONDS": 30.0,
    "MAX_TEST_UTTERANCES": None,
    "LONGFORM_BC_EDGE_TRIM_SECONDS": 2.0,
    "BC_SIGMA": 0.3,
    "BC_ORDER": 7,
    "EMAGE_EVALTOOLS_REVISION": "62450129f5a5b1fe5ea72fe9a2d762072fb861e7",
    "MIBURI_ASSET_REVISION": "188999d3860c81e86c5585a15d747594a8d60394",
    # Data paths, runtime, optimization, and evaluation settings.
    "SNAPSHOT_ROOT": "/kaggle/input/datasets/jackrichmaya/mybeatx-voxtream2-miburi-snapshot-v1",
    "FEATURE_ROOT": "/kaggle/input/datasets/jackrichmaya/mybeatx-voxtream2-miburi-snapshot-v1/features/tt_layer_11/942219f5359aec14",
    "CANONICAL_ROOT": "",
    "MANIFEST_ROOT": "",
    "RAW_BEATX_ROOT": "/kaggle/input/datasets/soheilkhalesy/mybeat2/BEAT2/beat_english_v2.0.0",
    "OUTPUT_DIR": "./runs/voxmove",
    "RESUME_PATH": "",
    "CHECKPOINT_PATH": "",
    "INIT_ADAPTED_PATH": "",
    "AUTO_RESUME": True,
    "RUN_VALIDATION": False,
    "NUM_GPUS": 1,
    "EVAL_CPU_THREADS": 1,
    "TRAIN_CPU_THREADS": 1,
    "USE_CUDA_GRAPHS": True,
    "CUDA_GRAPH_VERIFY": True,
    "CUDA_GRAPH_ON_FAILURE": "error",
    "EVAL_TIMING_FIRST_N": 2,
    "TT_LAYER": 11,
    "TT_DIM": 1024,
    "NUM_SPEAKERS": 31,
    "USE_ADDITIONAL": True,
    "EXPR_STD_FLOOR": 0.05,
    "FLOW_STEPS": 10,
    "CFG_DROPOUT": 0.1,
    "CFG_SCALE": 3.0,
    "GEOMETRIC_LOSS_WEIGHT": 1.0,
    "STRUCTURE_LOSS_WEIGHT": 1.0,
    "EXPRESSION_LOSS_WEIGHT": 1.0,
    "TEMPORAL_CONTRASTIVE_WEIGHT": 0.1,
    "SEED": 2342,
    "EVAL_SEED": 7331,
    "MIXED_PRECISION": "fp16",
    "FP16_GRAD_SCALER": {"init_scale": 2**10, "growth_factor": 1.5, "backoff_factor": 0.25, "growth_interval": 10000},
    "GRADIENT_CHECKPOINTING": False,
    "BATCH_SIZE_PER_GPU": 128,
    "GRADIENT_ACCUMULATION": 2,
    "NUM_WORKERS": 8,
    "LEARNING_RATE": 0.0001,
    "USE_FUSED_ADAM": True,
    "LR_SCHEDULER": "cosine",
    "COSINE_MIN_LR": 2e-06,
    "COSINE_T_MAX_UPDATES": None,
    "MAX_GRAD_NORM": 1.0,
    "EPOCHS": 500,
    "MAX_TRAIN_HOURS": None,
    "MAX_UPDATES": None,
    "LOG_EVERY_UPDATES": 50,
    "SAVE_EVERY_UPDATES": 0,
    "SAVE_BEST_VAL_LOSS": False,
    "VAL_LOSS_MAX_BATCHES": 64,
    "VAL_FGD_EVERY_EPOCHS": 5,
    "VAL_FGD_NUM_UTTERANCES": 32,
    "VAL_FGD_SUBSET_SEED": 2342,
    "FULL_TEST_CHECKPOINT": "last",
    "SMPLX_CHUNK_SIZE": 64,
    "SAVE_TEST_PREDICTIONS": True,
}
ARCHITECTURE_KEYS = [
    "MODEL_WIDTH",
    "MODEL_LAYERS",
    "NUM_SPEAKERS",
    "TT_LAYER",
    "TT_DIM",
    "USE_TT_HIDDEN",
    "USE_VOXTREAM_AUDIO_TOKENS",
    "USE_VOXTREAM_TEXT_TOKENS",
    "AUDIO_VOCAB",
    "PHONE_VOCAB",
    "FFN_VARIANT",
    "SWIGLU_HIDDEN_DIM",
]


def validate_config(cfg):
    # Only enforce genuine implementation constraints; checkpoint architecture is checked separately.
    if not isinstance(cfg["MODEL_WIDTH"], int) or cfg["MODEL_WIDTH"] <= 64 or cfg["MODEL_WIDTH"] % 24:
        raise ValueError("MODEL_WIDTH must be an integer >64 divisible by 24 (8 attention heads and SwiGLU)")
    if not isinstance(cfg["MODEL_LAYERS"], int) or cfg["MODEL_LAYERS"] < 1:
        raise ValueError("MODEL_LAYERS must be a positive integer")
    if not isinstance(cfg["WINDOW_FRAMES"], int) or cfg["WINDOW_FRAMES"] < 2 or cfg["WINDOW_FRAMES"] % 2:
        raise ValueError("WINDOW_FRAMES must be a positive even integer (25Hz motion / 12.5Hz speech)")
    if not isinstance(cfg["HISTORY_FRAMES"], int) or not 0 < cfg["HISTORY_FRAMES"] < cfg["WINDOW_FRAMES"]:
        raise ValueError("HISTORY_FRAMES must be between 1 and WINDOW_FRAMES-1")
    if not isinstance(cfg["TRAIN_STRIDE_FRAMES"], int) or cfg["TRAIN_STRIDE_FRAMES"] < 1:
        raise ValueError("TRAIN_STRIDE_FRAMES must be a positive integer")
    if cfg["MIXED_PRECISION"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("MIXED_PRECISION must be bf16, fp16 or fp32")
    scaler = cfg["FP16_GRAD_SCALER"]
    if not isinstance(scaler, dict) or set(scaler) != {"init_scale", "growth_factor", "backoff_factor", "growth_interval"}:
        raise ValueError("FP16_GRAD_SCALER must contain the four GradScaler parameters")
    if not all(isinstance(scaler[k], (float, int)) and not isinstance(scaler[k], bool) and math.isfinite(scaler[k]) for k in ("init_scale", "growth_factor", "backoff_factor")):
        raise ValueError("FP16_GRAD_SCALER has invalid numeric settings")
    if scaler["init_scale"] <= 0 or scaler["growth_factor"] <= 1 or not 0 < scaler["backoff_factor"] < 1:
        raise ValueError("FP16_GRAD_SCALER scale factors are invalid")
    if not isinstance(scaler["growth_interval"], int) or isinstance(scaler["growth_interval"], bool) or scaler["growth_interval"] < 1:
        raise ValueError("FP16_GRAD_SCALER growth_interval must be a positive integer")
    for key in (
        "NUM_SPEAKERS",
        "TT_DIM",
        "BATCH_SIZE_PER_GPU",
        "GRADIENT_ACCUMULATION",
        "FLOW_STEPS",
        "NUM_GPUS",
        "TRAIN_CPU_THREADS",
        "EVAL_CPU_THREADS",
        "EPOCHS",
        "LOG_EVERY_UPDATES",
        "VAL_LOSS_MAX_BATCHES",
        "VAL_FGD_EVERY_EPOCHS",
        "VAL_FGD_NUM_UTTERANCES",
        "SMPLX_CHUNK_SIZE",
    ):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["NUM_WORKERS"] < 0 or cfg["EVAL_TIMING_FIRST_N"] < 0:
        raise ValueError("Worker and timing counts must be nonnegative")
    if cfg["RESUME_PATH"] and cfg["INIT_ADAPTED_PATH"]:
        raise ValueError("Choose RESUME_PATH or INIT_ADAPTED_PATH")
    if not 0 <= cfg["CFG_DROPOUT"] <= 1:
        raise ValueError("CFG_DROPOUT must be between zero and one")
    for key in (
        "GEOMETRIC_LOSS_WEIGHT",
        "STRUCTURE_LOSS_WEIGHT",
        "EXPRESSION_LOSS_WEIGHT",
        "TEMPORAL_CONTRASTIVE_WEIGHT",
        "CFG_SCALE",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("LEARNING_RATE", "MAX_GRAD_NORM", "EXPR_STD_FLOOR"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["LR_SCHEDULER"] not in ("none", "cosine") or not 0 <= cfg["COSINE_MIN_LR"] < cfg["LEARNING_RATE"]:
        raise ValueError("Invalid learning-rate schedule")
    for key in ("COSINE_T_MAX_UPDATES", "MAX_UPDATES", "MAX_TRAIN_HOURS"):
        if cfg[key] is not None and cfg[key] <= 0:
            raise ValueError(f"{key} must be positive or null")
    if cfg["SAVE_EVERY_UPDATES"] is not None and cfg["SAVE_EVERY_UPDATES"] < 0:
        raise ValueError("SAVE_EVERY_UPDATES must be nonnegative or null")
    if cfg["CUDA_GRAPH_ON_FAILURE"] not in ("error", "eager"):
        raise ValueError("CUDA_GRAPH_ON_FAILURE must be error or eager")


def load_config(path):
    path = Path(path).resolve()
    overrides = json.loads(path.read_text())
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    for key in (
        "SNAPSHOT_ROOT",
        "FEATURE_ROOT",
        "CANONICAL_ROOT",
        "MANIFEST_ROOT",
        "RAW_BEATX_ROOT",
        "OUTPUT_DIR",
        "RESUME_PATH",
        "CHECKPOINT_PATH",
        "INIT_ADAPTED_PATH",
    ):
        if cfg[key] and (not Path(cfg[key]).is_absolute()):
            cfg[key] = str((path.parent / cfg[key]).resolve())
    validate_config(cfg)
    return cfg
