import os, sys, io, contextlib, hashlib, json, shutil, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import linalg
from scipy.spatial.transform import Rotation as SciRotation
from huggingface_hub import hf_hub_download, snapshot_download


def prepare_evaluation_assets(cfg):
    root = Path(cfg["OUTPUT_DIR"]) / "dependencies" / "emage_evaltools"
    needed = [
        "__init__.py",
        "mertic.py",
        "motion_encoder.py",
        "skeleton.py",
        "skeleton_DME.py",
        "decoders.py",
        "mean_vel_smplxflame_30.npy",
        "AESKConv_240_100.bin",
    ]
    stamp = root / "pinned_revision.txt"
    revision = cfg["EMAGE_EVALTOOLS_REVISION"]
    if (
        not stamp.is_file()
        or stamp.read_text().strip() != revision
        or any((not (root / f).is_file() for f in needed))
    ):
        snapshot_download(
            "H-Liu1997/emage_evaltools", revision=revision, local_dir=str(root), allow_patterns=needed
        )
        stamp.write_text(revision)
    target = root / "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
    expected = "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
    receipt = target.with_suffix(".verified.json")
    verified = False
    if target.is_file() and receipt.is_file():
        check = json.loads(receipt.read_text())
        verified = check == {
            "sha256": expected,
            "size": target.stat().st_size,
            "mtime_ns": target.stat().st_mtime_ns,
        }
    if not verified:
        if not target.is_file() or file_sha256(target) != expected:
            cached = hf_hub_download(
                "m-hamza-mughal/miburi-release-assets",
                "smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz",
                revision=cfg["MIBURI_ASSET_REVISION"],
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)
        if file_sha256(target) != expected:
            raise RuntimeError("Official SMPL-X asset checksum mismatch")
        receipt.write_text(
            json.dumps(
                {"sha256": expected, "size": target.stat().st_size, "mtime_ns": target.stat().st_mtime_ns}
            )
        )
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    importlib.invalidate_caches()
    return root


@contextlib.contextmanager
def mute_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def stable_seed(text, base):
    digest = hashlib.sha256(f"{base}|{text}".encode()).digest()[:8]
    return int.from_bytes(digest, "big") % (2**63 - 1)


def discover_raw_beat_root(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = list(Path("/kaggle/input").glob("**/beat_english_v2.0.0"))
    valid = [
        path.resolve()
        for path in candidates
        if (path / "wave16k").is_dir()
        and (path / "smplxflame_30").is_dir()
        and (path / "train_test_split.csv").is_file()
    ]
    valid = sorted(set(valid))
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected one BEAT2 root containing wave16k/, smplxflame_30/, and train_test_split.csv; found {valid}. Set RAW_BEATX_ROOT."
        )
    return valid[0]


def quat_slerp(q0, q1, a):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = dot > 0.9995
    s0 = np.sin((1.0 - a) * theta) / np.maximum(sin_theta, 1e-08)
    s1 = np.sin(a * theta) / np.maximum(sin_theta, 1e-08)
    out = s0 * q0 + s1 * q1
    linear = (1.0 - a) * q0 + a * q1
    out = np.where(near, linear, out)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-08)


def resample_pose_slerp(pose25, source_fps=25.0, target_fps=30.0):
    pose25 = np.asarray(pose25, np.float32).reshape(-1, 55, 3)
    if len(pose25) <= 1:
        return pose25.copy()
    n = max(1, int(round(len(pose25) * float(target_fps) / float(source_fps))))
    q = SciRotation.from_rotvec(pose25.reshape(-1, 3)).as_quat().reshape(len(pose25), 55, 4)
    pos = np.arange(n, dtype=np.float64) * float(source_fps) / float(target_fps)
    i0 = np.minimum(np.floor(pos).astype(np.int64), len(pose25) - 1)
    i1 = np.minimum(i0 + 1, len(pose25) - 1)
    a = (pos - i0).astype(np.float32)[:, None, None]
    qo = quat_slerp(q[i0], q[i1], a)
    return SciRotation.from_quat(qo.reshape(-1, 4)).as_rotvec().reshape(n, 55, 3).astype(np.float32)


def axis_angle_to_6d(pose):
    matrices = SciRotation.from_rotvec(np.asarray(pose).reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
    return matrices[:, :, :2, :].reshape(len(pose), 330).astype(np.float32)


def normalize_betas(value):
    result = np.zeros(300, dtype=np.float32)
    if value is not None:
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        result[: min(300, len(flat))] = flat[:300]
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_body_model(cfg, device):
    eval_root = prepare_evaluation_assets(cfg)
    import smplx

    with mute_output():
        body_model = (
            smplx.create(
                str(eval_root / "smplx_models"),
                model_type="smplx",
                gender="NEUTRAL_2020",
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                use_pca=False,
                ext="npz",
            )
            .to(device)
            .eval()
        )
    return body_model


def prepare_fgd_tool(cfg, device):
    eval_root = prepare_evaluation_assets(cfg)
    from emage_evaltools.mertic import FGD

    original_load = torch.load
    try:

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            kwargs.setdefault("map_location", device)
            return original_load(*args, **kwargs)

        torch.load = compatible_load
        with mute_output():
            fgd = FGD(download_path=str(eval_root) + os.sep, device=str(device))
    finally:
        torch.load = original_load
    return fgd


@torch.inference_mode()
def smplx_joints(toolbox, poses, betas, device, chunk=64):
    lengths = [len(pose) for pose in poses]
    pose = np.concatenate(poses, axis=0).reshape(-1, 165).astype(np.float32)
    beta = torch.from_numpy(normalize_betas(betas))[None].to(device).float()
    outputs = []
    for start in range(0, len(pose), chunk):
        p = torch.from_numpy(pose[start : start + chunk]).to(device).float()
        batch = len(p)
        zeros3 = torch.zeros((batch, 3), device=device)
        zeros100 = torch.zeros((batch, 100), device=device)
        result = toolbox["body_model"](
            betas=beta.expand(batch, -1),
            transl=zeros3,
            expression=zeros100,
            global_orient=p[:, :3],
            body_pose=p[:, 3:66],
            jaw_pose=p[:, 66:69],
            leye_pose=p[:, 69:72],
            reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120],
            right_hand_pose=p[:, 120:165],
            return_joints=True,
        )
        outputs.append(result.joints[:, :55].cpu().numpy())
    all_joints = np.concatenate(outputs, axis=0).astype(np.float32)
    result, cursor = ([], 0)
    for length in lengths:
        result.append(all_joints[cursor : cursor + length])
        cursor += length
    return result


def frechet_distance(left, right, eps=1e-06):
    left, right = (np.asarray(left), np.asarray(right))
    if len(left) <= 1 or len(right) <= 1:
        return float("nan")
    mu1, mu2 = (np.mean(left, axis=0), np.mean(right, axis=0))
    sigma1 = np.cov(left, rowvar=False)
    sigma2 = np.cov(right, rowvar=False)
    difference = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covariance_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covariance_mean)
    )


def resolve_raw_indices(raw_root):
    raw_root = Path(raw_root)
    motion_dir = raw_root / "smplxflame_30"
    audio_dir = raw_root / "wave16k"
    if not motion_dir.is_dir() or not audio_dir.is_dir():
        raise RuntimeError(f"LiveGesture/EMAGE-compatible evaluation requires {motion_dir} and {audio_dir}")
    motion = {path.stem: path for path in motion_dir.rglob("*.npz")}
    audio = {path.stem: path for path in audio_dir.rglob("*.wav")}
    if not motion or not audio:
        raise RuntimeError("Raw BEAT2 30-fps motion/audio index is empty")
    return (motion, audio)


def load_native_gt30(path):
    with np.load(path, allow_pickle=True) as archive:
        pose = None
        for key in ("poses", "pose", "motion"):
            if key in archive:
                pose = np.asarray(archive[key], dtype=np.float32)
                break
        if pose is None:
            raise KeyError(f"No poses/pose/motion array in {path}")
        pose = pose.reshape(-1, 55, 3)
        beta = None
        for key in ("betas", "beta", "shape"):
            if key in archive:
                beta = np.asarray(archive[key], dtype=np.float32).reshape(-1)
                break
    return (pose, normalize_betas(beta))


def official_test_ids(raw_root):
    split_path = Path(raw_root) / "train_test_split.csv"
    table = pd.read_csv(split_path)
    id_col = next((c for c in ("id", "utt_id", "utterance_id", "name") if c in table.columns), None)
    split_col = next((c for c in ("type", "split", "set") if c in table.columns), None)
    if id_col is None or split_col is None:
        raise RuntimeError(f"Cannot identify ID/split columns in {split_path}; columns={list(table.columns)}")
    mask = table[split_col].astype(str).str.strip().str.lower().eq("test")
    ids = set(table.loc[mask, id_col].astype(str))
    if not ids:
        raise RuntimeError(f"No test rows found in {split_path}")
    return (ids, split_path, len(table))


def longform_evaluation_manifest(cache, cfg, raw_root):
    manifest = cache.manifest.sort_values(["duration_seconds", "utt_id"], ascending=[True, True]).reset_index(
        drop=True
    )
    manifest = manifest.loc[manifest.duration_seconds >= float(cfg["LONGFORM_MIN_SECONDS"])].reset_index(
        drop=True
    )
    if cfg.get("MAX_TEST_UTTERANCES") is not None:
        manifest = manifest.iloc[: int(cfg["MAX_TEST_UTTERANCES"])].reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No test recordings are >= LONGFORM_MIN_SECONDS={cfg['LONGFORM_MIN_SECONDS']}")
    test_ids, split_path, split_rows = official_test_ids(raw_root)
    selected_ids = set(manifest.utt_id.astype(str))
    not_official_test = sorted(selected_ids - test_ids)
    if not_official_test:
        raise RuntimeError(
            f"Cached long-form manifest contains IDs not marked test in the official BEAT2 CSV: {not_official_test[:20]}"
        )
    return (
        manifest,
        {
            "split_csv": str(split_path),
            "split_csv_rows": int(split_rows),
            "official_test_ids": int(len(test_ids)),
            "selected_longform_ids": int(len(selected_ids)),
            "selected_ids_are_official_test": True,
        },
    )


def accumulator_state(metric):
    return {"sum": float(metric.sum), "counter": int(metric.counter)}


def concatenate_features(values):
    values = [np.asarray(value) for value in values if np.asarray(value).size]
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(values, axis=0)


def fgd_feature_arrays(fgd):
    pred_values = [np.asarray(x).reshape(-1, np.asarray(x).shape[-1]) for x in fgd.pred_features]
    gt_values = [np.asarray(x).reshape(-1, np.asarray(x).shape[-1]) for x in fgd.target_features]
    return (concatenate_features(pred_values), concatenate_features(gt_values))


def _validation_fgd_fixed_manifest(cache, cfg):
    manifest = cache.manifest.drop_duplicates("utt_id").copy()
    if "duration_seconds" in manifest.columns:
        manifest = manifest.loc[manifest["duration_seconds"].astype(float) >= 32.0 / 30.0]
    elif "common_frames_12p5" in manifest.columns:
        manifest = manifest.loc[manifest["common_frames_12p5"].astype(int) >= 14]
    if manifest.empty:
        raise RuntimeError("No validation utterances are long enough for one EMAGE FGD feature block")
    seed = int(cfg.get("VAL_FGD_SUBSET_SEED", cfg.get("SEED", 0)))
    manifest["_fixed_fgd_key"] = (
        manifest["utt_id"]
        .astype(str)
        .map(lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    )
    manifest = manifest.sort_values(["_fixed_fgd_key", "utt_id"]).drop(columns="_fixed_fgd_key")
    requested = cfg.get("VAL_FGD_NUM_UTTERANCES", None)
    if requested is not None:
        requested = int(requested)
        if requested < 1:
            raise ValueError("VAL_FGD_NUM_UTTERANCES must be >= 1 or None")
        manifest = manifest.iloc[:requested]
    return manifest.reset_index(drop=True)


def prepare_metric_tools(cfg, device):
    fgd = prepare_fgd_tool(cfg, device)
    from emage_evaltools.mertic import BC, L1div

    return {
        "BC": BC,
        "L1div": L1div,
        "fgd": fgd,
        "body_model": load_body_model(cfg, device),
        "root": Path(cfg["OUTPUT_DIR"]) / "dependencies/emage_evaltools",
    }
