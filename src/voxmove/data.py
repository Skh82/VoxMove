import os
import json
import hashlib
from pathlib import Path
from urllib.parse import quote
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from .geometry import D, part_indices, local_to_global
from .runtime import write_json
from . import metrics as official


def paths(cfg):
    root = Path(cfg["SNAPSHOT_ROOT"]).resolve()
    if not (root / "windows/train.parquet").is_file():
        raise FileNotFoundError(f"{root}/windows/train.parquet; set SNAPSHOT_ROOT to the existing snapshot")
    raw = official.discover_raw_beat_root(cfg["RAW_BEATX_ROOT"])
    return (root, raw, Path(cfg["OUTPUT_DIR"]))


def layer_dir(cfg, split):
    root, _, _ = paths(cfg)
    mapping = root / "features/hidden_cache_config.json"
    layer = str(cfg["TT_LAYER"])
    if mapping.is_file():
        mapping = json.loads(mapping.read_text())["layer_cache_relpaths"]
        if layer not in mapping:
            raise KeyError(f"TT layer {layer} missing; available={list(mapping)}")
        result = root / mapping[layer]
    else:
        result = Path(cfg["FEATURE_ROOT"])
    if not (result / f"{split}.h5").is_file():
        raise FileNotFoundError(result / f"{split}.h5")
    return result


def manifest(cfg, split):
    root, _, _ = paths(cfg)
    p = Path(cfg.get("MANIFEST_ROOT") or root / "manifests") / f"canonical_{split}.parquet"
    table = pd.read_parquet(p)
    required = {"utt_id", "h5_key", "common_frames_12p5", "duration_seconds"}
    if required - set(table):
        raise ValueError(f"{p} missing {required - set(table)}")
    if table.utt_id.duplicated().any():
        raise ValueError(f"Duplicate utterances in {p}")
    return table.sort_values("utt_id").reset_index(drop=True)


class Cache:
    def __init__(self, cfg, split):
        self.cfg, self.split = (cfg, split)
        self.root, self.raw, self.out = paths(cfg)
        self.manifest = manifest(cfg, split)
        windows = pd.read_parquet(self.root / f"windows/{split}.parquet")
        columns = ["utt_id", "speech_chunk_id", "speech_chunk_start", "speech_chunk_end"]
        self.chunks = {
            str(k): v[columns].drop_duplicates().sort_values("speech_chunk_start")
            for k, v in windows.groupby("utt_id", sort=False)
        }
        self.canonical_path = Path(cfg.get("CANONICAL_ROOT") or self.root / "canonical") / f"{split}.h5"
        self.hidden_path = layer_dir(cfg, split) / f"{split}.h5"
        self._hidden = None

    def close(self):
        for f in [self._hidden]:
            if f is not None:
                f.close()
        self._hidden = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hidden"] = None
        return state

    def load(self, row, start=0, end=None):
        if self._hidden is None:
            self._hidden = h5py.File(self.hidden_path, "r")
        uid = str(row.utt_id)
        total = int(row.common_frames_12p5)
        stop = total if end is None else min(int(end), total)
        start = int(start)
        n = stop - start
        if n < 1:
            raise ValueError(f"Empty utterance {uid}")
        hidden = np.zeros((n, self.cfg["TT_DIM"]), np.float32)
        covered = np.zeros(n, bool)
        seams = []
        if uid not in self.chunks:
            raise ValueError(f"No speech chunks for {uid}")
        for chunk in self.chunks[uid].itertuples():
            a, b = (int(chunk.speech_chunk_start), int(chunk.speech_chunk_end))
            key = quote(str(chunk.speech_chunk_id), safe="-_.")
            dataset = self._hidden[key]["hidden"]
            if dataset.ndim != 2 or dataset.shape[1] != hidden.shape[1] or len(dataset) < b - a:
                raise ValueError(f"{uid}: chunk {key} shape {dataset.shape}, span [{a},{b})")
            lo, hi = (max(a, start), min(b, stop))
            if hi <= lo:
                continue
            values = np.asarray(dataset[lo - a : hi - a], np.float32)
            lo -= start
            hi -= start
            if covered[lo:hi].any():
                mask = covered[lo:hi]
                if not np.allclose(hidden[lo:hi][mask], values[mask], rtol=0.0001, atol=0.0001):
                    raise ValueError(f"Conflicting overlapping hidden chunks: {uid}")
            hidden[lo:hi] = values
            covered[lo:hi] = True
            if 0 < lo < n:
                seams.append(lo)
        if not covered.all():
            first = int(np.flatnonzero(~covered)[0])
            if covered[first:].any():
                raise ValueError(f"Interior hidden-cache gap in {uid} at tick {first}")
            if first == 0:
                raise ValueError(f"No initial cache coverage for {uid}")
            hidden = hidden[:first]
        usable = len(hidden)
        result = {
            "hidden": hidden,
            "uid": uid,
            "pid": int(uid.split("_")[0]),
            "seams": np.asarray(sorted(set((s for s in seams if s < usable))), np.int64),
            "dropped_tail_ticks": n - usable,
        }
        if not 0 <= result["pid"] < self.cfg["NUM_SPEAKERS"]:
            raise ValueError(f"Unsupported speaker ID: {uid}")
        if not np.isfinite(hidden).all():
            raise ValueError(f"Nonfinite hidden cache: {uid}")
        return result


def split_audit(cfg):
    root, raw, out = paths(cfg)
    csv = pd.read_csv(raw / "train_test_split.csv")
    idcol = next((c for c in ["id", "utt_id", "utterance_id", "name"] if c in csv))
    typecol = next((c for c in ["type", "split", "set"] if c in csv))
    official_sets = {str(k).lower().strip(): set(v[idcol].astype(str)) for k, v in csv.groupby(typecol)}
    report = {}
    seen = set()
    splits = ["train", "val", "test"] + (["additional"] if cfg["USE_ADDITIONAL"] else [])
    for split in splits:
        table = manifest(cfg, split)
        ids = set(table.utt_id.astype(str))
        if seen & ids:
            raise ValueError(f"Split overlap: {sorted(seen & ids)[:5]}")
        seen |= ids
        allowed = official_sets.get(
            split, official_sets.get("validation", set()) if split == "val" else set()
        )
        if ids - allowed:
            raise ValueError(
                f"{split}: {len(ids - allowed)} IDs not in the official split, e.g. {sorted(ids - allowed)[:5]}"
            )
        report[split] = {
            "utterances": len(table),
            "hours": float(table.duration_seconds.sum() / 3600),
            "id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        }
    write_json(out / "split_audit.json", report)
    return report


def data_signature(cfg):
    root, raw, _ = paths(cfg)
    keys = [
        "TT_LAYER",
        "TT_DIM",
        "USE_ADDITIONAL",
        "USE_VOXTREAM_AUDIO_TOKENS",
        "USE_VOXTREAM_TEXT_TOKENS",
        "PHONE_VOCAB",
        "AUDIO_VOCAB",
        "EXPR_STD_FLOOR",
    ]
    payload = {"format": 5, "config": {k: cfg[k] for k in keys}, "snapshot": str(root), "raw": str(raw)}
    files = [raw / "train_test_split.csv"]
    for s in ["train", "val", "test"] + (["additional"] if cfg["USE_ADDITIONAL"] else []):
        c = Cache(cfg, s)
        files += [
            Path(cfg.get("MANIFEST_ROOT") or root / "manifests") / f"canonical_{s}.parquet",
            root / f"windows/{s}.parquet",
            c.canonical_path,
            c.hidden_path,
        ]
    payload["files"] = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def linear_resample(x, n, source=30.0, target=25.0):
    x = np.asarray(x, np.float32)
    t = np.arange(n) * source / target
    i = np.minimum(t.astype(np.int64), len(x) - 1)
    j = np.minimum(i + 1, len(x) - 1)
    a = (t - i).clip(0, 1).reshape((-1,) + (1,) * (x.ndim - 1))
    return ((1 - a) * x[i] + a * x[j]).astype(np.float32)


def prepare_data(cfg):
    _, raw, out = paths(cfg)
    report = split_audit(cfg)
    sig = data_signature(cfg)
    dest = out / "prepared" / sig
    receipt = dest / "complete.json"
    if receipt.is_file():
        if all(((dest / f"{s}.h5").is_file() for s in report)) and (dest / "stats.npz").is_file():
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    parts, groups = part_indices(cfg)
    sums = np.zeros(100, np.float64)
    sq = np.zeros(100, np.float64)
    count = 0
    toolbox = official.load_body_model(cfg, "cpu").requires_grad_(False)
    rest_cache = {}
    details = {}
    raw_index = {p.stem: p for p in (raw / "smplxflame_30").rglob("*.npz")}
    for split in report:
        cache = Cache(cfg, split)
        nframes = 0
        drops = 0
        tmp = dest / f"{split}.h5.tmp"
        with h5py.File(tmp, "w") as target, h5py.File(cache.canonical_path, "r") as canonical:
            for row in tqdm(cache.manifest.itertuples(), total=len(cache.manifest), desc=f"Prepare {split}"):
                speech = cache.load(row)
                n = 2 * len(speech["hidden"])
                uid = str(row.utt_id)
                if uid not in raw_index:
                    raise FileNotFoundError(f"Raw SMPL-X targets missing: {uid}")
                with np.load(raw_index[uid], allow_pickle=True) as data:
                    pose = np.asarray(data["poses"], np.float32).reshape(-1, 55, 3)
                    expr = np.asarray(data["expressions"], np.float32)[:, :100]
                    trans = np.asarray(data["trans"], np.float32)
                    betas = official.normalize_betas(data["betas"])
                    fps = float(np.asarray(data.get("mocap_frame_rate", 30.0)).reshape(-1)[0])
                if abs(fps - 30) > 0.001:
                    raise ValueError(f"{uid}: native evaluation assumes 30 FPS, found {fps}")
                if min(len(pose), len(expr), len(trans)) / 30 < n / 25 - 1 / 25:
                    raise ValueError(f"{uid}: speech cache is longer than raw motion")
                pose25 = official.resample_pose_slerp(pose, 30.0, 25.0)
                if len(pose25) < n:
                    pose25 = np.concatenate([pose25, np.repeat(pose25[-1:], n - len(pose25), 0)])
                local = Rotation.from_rotvec(pose25[:n].reshape(-1, 3)).as_matrix().reshape(n, 55, 3, 3)
                global_r6 = local_to_global(local)[..., :2, :].reshape(n, 330).astype(np.float32)
                expression = linear_resample(expr, n)
                translation = linear_resample(trans, n)
                x = np.concatenate([global_r6, translation, expression], -1)
                if not np.isfinite(x).all():
                    raise ValueError(f"Nonfinite motion targets: {uid}")
                source = canonical[str(row.h5_key)]
                valid = np.ones((n, D), np.float32)
                for name, idx in [
                    ("motion_valid12p5", np.arange(D)),
                    ("lower_valid12p5", np.r_[groups["lower"], np.arange(330, 333)]),
                    ("face_valid12p5", parts["face"]),
                ]:
                    if name not in source:
                        raise KeyError(f"{uid}: missing mandatory target mask {name}")
                    mask = np.asarray(source[name][: n // 2], bool).reshape(-1)
                    if len(mask) * 2 != n:
                        raise ValueError(f"{uid}: short {name}")
                    valid[:, idx] *= np.repeat(mask, 2)[:, None]
                key = quote(uid, safe="-_.")
                g = target.create_group(key)
                for k, a in [("x", x), ("valid", valid.astype(np.uint8)), ("seams", speech["seams"])]:
                    g.create_dataset(k, data=a, compression="lzf" if len(a) > 0 else None)
                rest_key = hashlib.sha256(betas.tobytes()).hexdigest()
                if rest_key not in rest_cache:
                    with torch.inference_mode():
                        rest_cache[rest_key] = (
                            toolbox(betas=torch.from_numpy(betas)[None], return_verts=False)
                            .joints[0, :55]
                            .numpy()
                            .astype(np.float32)
                        )
                g.create_dataset("rest", data=rest_cache[rest_key])
                g.create_dataset("betas", data=betas)
                g.attrs["pid"] = speech["pid"]
                g.attrs["dropped_tail_ticks"] = speech["dropped_tail_ticks"]
                nframes += n
                drops += speech["dropped_tail_ticks"]
                if split in ["train", "additional"]:
                    good = valid[:, 333] > 0
                    sums += expression[good].sum(0, dtype=np.float64)
                    sq += (expression[good].astype(np.float64) ** 2).sum(0)
                    count += int(good.sum())
        cache.close()
        os.replace(tmp, dest / f"{split}.h5")
        details[split] = {
            "frames25": nframes,
            "usable_hours": nframes / 25 / 3600,
            "dropped_tail_ticks": drops,
        }
    if count < 2:
        raise ValueError("No valid training expressions")
    mean = sums / count
    std = np.sqrt(np.maximum(sq / count - mean**2, 0)).clip(cfg["EXPR_STD_FLOOR"])
    np.savez(
        dest / "stats.npz", expr_mean=mean.astype(np.float32), expr_std=std.astype(np.float32), count=count
    )
    write_json(
        receipt,
        {
            "signature": sig,
            "splits": details,
            "stats_source": ["train"] + (["additional"] if cfg["USE_ADDITIONAL"] else []),
        },
    )
    return dest


def normalize_state(x, stats):
    x = x.copy()
    x[..., 333:] = (x[..., 333:] - stats["expr_mean"]) / stats["expr_std"]
    return x


class Windows(torch.utils.data.Dataset):
    def __init__(self, cfg, prepared, splits):
        self.cfg = cfg
        self.prepared = Path(prepared)
        self.handles = {}
        self.rows = []
        self.caches = {s: Cache(cfg, s) for s in splits}
        self.metadata = {
            s: {quote(str(r.utt_id), safe="-_."): r for r in self.caches[s].manifest.itertuples()}
            for s in splits
        }
        with np.load(self.prepared / "stats.npz") as f:
            self.stats = {k: f[k] for k in ["expr_mean", "expr_std"]}
        for split in splits:
            with h5py.File(self.prepared / f"{split}.h5", "r") as f:
                for uid in sorted(f):
                    n = len(f[uid]["x"])
                    L = cfg["WINDOW_FRAMES"]
                    if n < L:
                        continue
                    starts = list(range(0, n - L + 1, cfg["TRAIN_STRIDE_FRAMES"]))
                    if starts[-1] != n - L:
                        starts.append((n - L) // 2 * 2)
                    self.rows.extend(((split, uid, s) for s in sorted(set(starts))))
        if not self.rows:
            raise ValueError("No complete training/validation windows")

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["handles"] = {}
        return d

    def __getitem__(self, i):
        split, uid, start = self.rows[i]
        L = self.cfg["WINDOW_FRAMES"]
        if split not in self.handles:
            self.handles[split] = h5py.File(self.prepared / f"{split}.h5", "r")
        g = self.handles[split][uid]
        x = normalize_state(np.asarray(g["x"][start : start + L]), self.stats)
        x[:, 330:333] -= x[0, 330:333].copy()
        valid = np.asarray(g["valid"][start : start + L])
        neutral = np.zeros(D, np.float32)
        neutral[:330] = np.tile([1, 0, 0, 0, 1, 0], 55)
        x = np.where(valid > 0, x, neutral)
        speech = self.caches[split].load(self.metadata[split][uid], start // 2, (start + L) // 2)
        if len(speech["hidden"]) != L // 2:
            raise ValueError(f"{uid}: cropped cache unexpectedly short")
        result = {
            "x": x,
            "valid": valid.astype(np.float32),
            "hidden": speech["hidden"],
            "pid": np.asarray(g.attrs["pid"], np.int64),
            "rest": np.asarray(g["rest"]),
        }
        return {k: torch.from_numpy(np.asarray(v)) for k, v in result.items()}
