import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from .data import paths, Cache
from .checkpoint import load_adapted
from .runtime import stats_from, sampler_identity, validation_identity, barrier, write_json
from .generate import generate, CFGPredictor
from . import metrics as official


def read_shape(prepared, split, uid):
    from urllib.parse import quote

    with h5py.File(Path(prepared) / f"{split}.h5", "r") as f:
        g = f[quote(uid, safe="-_.")]
        return (np.asarray(g["rest"]), np.asarray(g["betas"]))


@torch.inference_mode()
def validation_fgd(model, cfg, prepared, device, should_stop=None):
    cache = Cache(cfg, "val")
    selected = official._validation_fgd_fixed_manifest(cache, cfg)
    stats = stats_from(prepared)
    fgd = official.prepare_fgd_tool(cfg, device)
    _, raw, _ = paths(cfg)
    motion, _ = official.resolve_raw_indices(raw)
    ids = []
    frames = 0
    was_training = model.training
    model.eval()
    engine = CFGPredictor(model, cfg, device)
    started = time.perf_counter()
    try:
        for row in tqdm(selected.itertuples(), total=len(selected), desc="Fixed validation FGD"):
            sample = cache.load(row)
            result = generate(model, sample, stats, cfg, device, predictor=engine, should_stop=should_stop)
            pred = official.resample_pose_slerp(result["poses"], 25.0, 30.0)
            gt, _ = official.load_native_gt30(motion[str(row.utt_id)])
            n = min(len(pred), len(gt))
            if n < 32:
                raise ValueError(f"{row.utt_id}: fewer than 32 aligned validation frames")
            p = torch.from_numpy(official.axis_angle_to_6d(pred[:n]))[None].to(device)
            q = torch.from_numpy(official.axis_angle_to_6d(gt[:n]))[None].to(device)
            with official.mute_output():
                fgd.update(p, q)
            ids.append(str(row.utt_id))
            frames += n
        p, q = official.fgd_feature_arrays(fgd)
        value = official.frechet_distance(p, q)
        if not np.isfinite(value):
            raise FloatingPointError("Validation FGD is nonfinite")
        return {
            "FGD_raw": value,
            "FGD_x10": 10 * value,
            "utterances": ids,
            "frames30": frames,
            "feature_rows": len(p),
            "seed": cfg["EVAL_SEED"],
            "sampler": sampler_identity(cfg),
            "selection": validation_identity(cfg),
            "inference": engine.report(),
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        engine.close()
        cache.close()
        model.train(was_training)


@torch.inference_mode()
def evaluate(cfg, prepared, rank, world, device, checkpoint_path):
    model, state = load_adapted(cfg, checkpoint_path, device, prepared)
    model.eval()
    if state["progress"]["selected_fgd_sampler"] != sampler_identity(cfg):
        raise ValueError(
            "Test sampler differs from validation checkpoint selection. Validate this sampler first in a separate run."
        )
    del state
    cache = Cache(cfg, "test")
    _, raw, out = paths(cfg)
    selected, split_info = official.longform_evaluation_manifest(cache, cfg, raw)
    ids = selected.utt_id.astype(str).tolist()
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    run_id = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(run_id, src=0)
    run = out / "evaluation" / (Path(checkpoint_path).stem + "_" + run_id[0])
    run.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        selected.to_csv(run / "selected_manifest.csv", index=False)
        temporal_protocol = {
            "temporal_mode": "causal",
            "history_seed": "16-frame neutral identity BOS",
            "generation_schedule": "50-frame windows = 16 clean history + 34 newly generated frames",
            "parameterization": "original clean-x0 flow sampler with FLOW_STEPS",
        }
        write_json(
            run / "protocol.json",
            {
                "split": split_info,
                "manifest_sha256": digest,
                "EMAGE_revision": cfg["EMAGE_EVALTOOLS_REVISION"],
                "resampling": "25->30 quaternion SLERP, exact reference helper",
                "FGD_input": "55 local joint rotations, 330 dimensions, first two matrix rows",
                "FGD_aggregation": "full-utterance official features pooled over every worker; raw Frechet once",
                "BC_DIV_input": "official 55 SMPL-X joints; zero translation and expressions; GT identity betas",
                **temporal_protocol,
                "sampler": sampler_identity(cfg),
                "face_expressions_scored_by_official_metrics": False,
            },
        )
    toolbox = official.prepare_metric_tools(cfg, device)
    fgd = toolbox["fgd"]
    with official.mute_output():
        bc = toolbox["BC"](
            download_path=str(toolbox["root"]) + os.sep, sigma=cfg["BC_SIGMA"], order=cfg["BC_ORDER"]
        )
        bc_gt = toolbox["BC"](
            download_path=str(toolbox["root"]) + os.sep, sigma=cfg["BC_SIGMA"], order=cfg["BC_ORDER"]
        )
    div = toolbox["L1div"]()
    div_gt = toolbox["L1div"]()
    motion, audio = official.resolve_raw_indices(raw)
    stats = stats_from(prepared)
    processed = []
    frames = 0
    bc_samples = 0
    dropped = 0
    generation_seconds = 0.0
    content_seconds = 0.0
    seam_jumps = []
    engine = CFGPredictor(model, cfg, device)

    def clock():
        if timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    started = time.perf_counter()
    try:
        for row in tqdm(
            selected.iloc[rank::world].itertuples(),
            total=len(selected.iloc[rank::world]),
            desc=f"Official test rank {rank}",
        ):
            uid = str(row.utt_id)
            if uid not in motion or uid not in audio:
                raise FileNotFoundError(f"Missing native motion/audio: {uid}")
            timing = len(processed) < cfg["EVAL_TIMING_FIRST_N"]
            t0 = clock()
            sample = cache.load(row)
            _, betas = read_shape(prepared, "test", uid)
            t1 = clock()
            result = generate(model, sample, stats, cfg, device, predictor=engine)
            t2 = clock()
            generation_seconds += result["generation_seconds"]
            content_seconds += len(result["poses"]) / 25
            dropped += sample["dropped_tail_ticks"]
            if cfg["SAVE_TEST_PREDICTIONS"]:
                pred_dir = run / "predictions"
                pred_dir.mkdir(exist_ok=True)
                payload = dict(
                    poses=result["poses"].reshape(-1, 165),
                    trans=result["trans"],
                    expressions=result["expressions"],
                    betas=betas,
                    mocap_frame_rate=25,
                    model="smplx",
                    gender="neutral",
                    window_seams25=result["window_seams25"],
                    cache_seams12p5=result["cache_seams12p5"],
                )
                np.savez_compressed(pred_dir / f"{uid}.npz", **payload)
            pred = official.resample_pose_slerp(result["poses"], 25.0, 30.0)
            gt, betas = official.load_native_gt30(motion[uid])
            n = min(len(pred), len(gt))
            if n < 32:
                raise ValueError(f"{uid}: fewer than 32 aligned frames")
            pred = pred[:n]
            gt = gt[:n]
            frames += n
            p = torch.from_numpy(official.axis_angle_to_6d(pred))[None].to(device)
            q = torch.from_numpy(official.axis_angle_to_6d(gt))[None].to(device)
            t3 = clock()
            with official.mute_output():
                fgd.update(p, q)
            t4 = clock()
            pj, gj = official.smplx_joints(toolbox, [pred, gt], betas, device, cfg["SMPLX_CHUNK_SIZE"])
            t5 = clock()
            pf = pj.reshape(n, -1)
            gf = gj.reshape(n, -1)
            div.compute(pf)
            div_gt.compute(gf)
            trim_seconds = cfg["LONGFORM_BC_EDGE_TRIM_SECONDS"]
            trim = round(trim_seconds * 30)
            if n > 2 * trim + 2 * cfg["BC_ORDER"] + 3:
                audio_beats = bc.load_audio(
                    str(audio[uid]), t_start=round(trim_seconds * 16000), t_end=round((n - trim) / 30 * 16000)
                )
                if len(audio_beats):
                    mb = bc.load_motion(pf, t_start=trim, t_end=n - trim, pose_fps=30, without_file=True)
                    gb = bc_gt.load_motion(gf, t_start=trim, t_end=n - trim, pose_fps=30, without_file=True)
                    bc.compute(audio_beats, mb, length=n - 2 * trim, pose_fps=30)
                    bc_gt.compute(audio_beats, gb, length=n - 2 * trim, pose_fps=30)
                    bc_samples += 1
            for seam in result["window_seams25"]:
                i = round(float(seam) / 25 * 30)
                if 0 < i < n:
                    seam_jumps.append(float(np.linalg.norm(pj[i] - pj[i - 1], axis=-1).mean()))
            if timing:
                t6 = clock()
                print(
                    "[EVAL_TIMING] "
                    + json.dumps(
                        {
                            "rank": rank,
                            "uid": uid,
                            "precision": cfg["MIXED_PRECISION"],
                            "motion_seconds": len(result["poses"]) / 25,
                            "load_s": t1 - t0,
                            "generation_s": t2 - t1,
                            "prepare_and_save_s": t3 - t2,
                            "fgd_features_s": t4 - t3,
                            "smplx_s": t5 - t4,
                            "bc_div_seams_s": t6 - t5,
                            "total_s": t6 - t0,
                            "graph_replays": engine.replays,
                        }
                    ),
                    flush=True,
                )
            processed.append(uid)
        features, targets = official.fgd_feature_arrays(fgd)
        np.savez(run / f"rank_{rank}_features.npz", pred=features, gt=targets)
        write_json(
            run / f"rank_{rank}.json",
            {
                "ids": processed,
                "manifest_sha256": digest,
                "frames30": frames,
                "BC": official.accumulator_state(bc),
                "BC_GT": official.accumulator_state(bc_gt),
                "DIV": official.accumulator_state(div),
                "DIV_GT": official.accumulator_state(div_gt),
                "bc_samples": bc_samples,
                "dropped_tail_ticks": dropped,
                "generation_seconds": generation_seconds,
                "content_seconds": content_seconds,
                "worker_wall_seconds": time.perf_counter() - started,
                "inference": engine.report(),
                "window_seam_jumps_m": seam_jumps,
            },
        )
    finally:
        engine.close()
        cache.close()
    barrier()
    if rank == 0:
        shards = [json.loads((run / f"rank_{i}.json").read_text()) for i in range(world)]
        actual = [uid for s in shards for uid in s["ids"]]
        if sorted(actual) != sorted(ids) or len(actual) != len(set(actual)):
            raise RuntimeError("Official test coverage mismatch or duplicate worker results")
        fs = []
        gs = []
        for i in range(world):
            with np.load(run / f"rank_{i}_features.npz") as f:
                if f["pred"].size:
                    fs.append(f["pred"])
                    gs.append(f["gt"])
        raw_fgd = official.frechet_distance(np.concatenate(fs), np.concatenate(gs))
        if not np.isfinite(raw_fgd):
            raise FloatingPointError("Nonfinite official full-test FGD")
        report = {
            "FGD_raw": raw_fgd,
            "FGD_x10": raw_fgd * 10,
            "utterances": len(ids),
            "manifest_sha256": digest,
            "evaluated_frames30": sum((s["frames30"] for s in shards)),
        }
        for key in ["BC", "BC_GT", "DIV", "DIV_GT"]:
            counter = sum((s[key]["counter"] for s in shards))
            total = sum((s[key]["sum"] for s in shards))
            report[key] = total / counter if counter else None
            report[key + "_counter"] = counter
        for key in ["bc_samples", "dropped_tail_ticks", "generation_seconds", "content_seconds"]:
            report[key] = sum((s[key] for s in shards))
        jumps = [x for s in shards for x in s["window_seam_jumps_m"]]
        report["diagnostic_window_seam_mean_m"] = float(np.mean(jumps)) if jumps else None
        report["BC_recordings_without_usable_beats"] = len(ids) - report["bc_samples"]
        report["checkpoint"] = str(checkpoint_path)
        report["sampler"] = sampler_identity(cfg)
        report["evaluation_workers"] = world
        report["inference_by_worker"] = [v["inference"] for v in shards]
        report["wall_seconds"] = time.perf_counter() - started
        write_json(run / "metrics.json", report)
        print(json.dumps(report, indent=2), flush=True)
        write_json(out / "evaluation/metrics.json", report)
        write_json(
            out / "evaluation/latest_run.json", {"run": str(run), "metrics": str(run / "metrics.json")}
        )
    barrier()
