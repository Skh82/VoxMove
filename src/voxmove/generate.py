import json
import torch
from .runtime import autocast_context


class CFGPredictor:
    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.enabled = bool(cfg["USE_CUDA_GRAPHS"] and self.device.type == "cuda")
        self.graph = None
        self.static = None
        self.output = None
        self.signature = None
        self.checked_phases = set()
        self.calls = self.replays = self.captures = 0
        self.failure = None

    def _eager(self, d):
        with autocast_context(self.cfg, self.device):
            args = (d["x"], d["t"], {"pid": d["pid"]}, d["p"], d["v"])
            conditional = self.model(*args, condition=d["c"])
            if self.cfg["CFG_SCALE"] == 1.0:
                return conditional
            unconditional = self.model(*args, condition=torch.zeros_like(d["c"]))
            return unconditional + self.cfg["CFG_SCALE"] * (conditional - unconditional)

    def _capture(self, d):
        self.static = {k: v.clone() for k, v in d.items()}
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.random.fork_rng(devices=[self.device.index]):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._eager(self.static)
            current.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._eager(self.static)
        self.captures += 1
        print(
            f"[CUDA_GRAPH] {self.device}: captured separate CFG; shape={tuple(d['x'].shape)}; precision={self.cfg['MIXED_PRECISION']}",
            flush=True,
        )

    @torch.inference_mode()
    def __call__(self, x, t, batch, prompt, visible, condition, phase):
        if self.model.training:
            raise RuntimeError("Inference graph called while model is in training mode")
        self.calls += 1
        d = dict(x=x, t=t, pid=batch["pid"], p=prompt, v=visible, c=condition)
        if not self.enabled:
            return self._eager(d)
        signature = tuple(((k, tuple(v.shape), v.dtype, v.device) for k, v in d.items()))
        try:
            if self.graph is None:
                self.signature = signature
                self._capture(d)
            elif signature != self.signature:
                raise ValueError("Graph input shape/dtype changed within a fixed-window evaluation pass")
            for k, v in d.items():
                self.static[k].copy_(v)
            self.graph.replay()
            self.replays += 1
            if self.cfg["CUDA_GRAPH_VERIFY"] and phase not in self.checked_phases:
                reference = self._eager(d)
                equal = torch.equal(self.output, reference)
                error = (self.output.float() - reference.float()).abs().max().item()
                print(
                    "[CUDA_GRAPH_CHECK] "
                    + json.dumps(
                        {
                            "device": str(self.device),
                            "phase": phase,
                            "exact_equal": equal,
                            "max_abs_error": error,
                        }
                    ),
                    flush=True,
                )
                if not equal:
                    raise RuntimeError(f"CUDA graph differs from eager in {phase}: max_abs={error}")
                self.checked_phases.add(phase)
            return self.output
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            self.graph = self.static = self.output = None
            if self.cfg["CUDA_GRAPH_ON_FAILURE"] == "error":
                raise RuntimeError(
                    'CUDA Graph path failed. See the cause below. Set USE_CUDA_GRAPHS=False for eager inference, or CUDA_GRAPH_ON_FAILURE="eager" for an explicit fallback.'
                ) from exc
            self.enabled = False
            print(f"[CUDA_GRAPH_FALLBACK] {self.device}: {self.failure}", flush=True)
            return self._eager(d)

    def report(self):
        return {
            "requested": self.cfg["USE_CUDA_GRAPHS"],
            "enabled": self.enabled,
            "device": str(self.device),
            "captures": self.captures,
            "calls": self.calls,
            "replays": self.replays,
            "verified_phases": sorted(self.checked_phases),
            "failure": self.failure,
        }

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize(self.device)
        self.graph = self.static = self.output = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


import time
import numpy as np
import torch
from .geometry import D, project_state, state_to_local_aa
from .metrics import stable_seed


class GenerationStopped(RuntimeError):
    pass


def neutral_history(length, device):
    frame = torch.zeros(D, dtype=torch.float32, device=device)
    frame[:330] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device).repeat(55)
    return frame[None].repeat(int(length), 1)


@torch.inference_mode()
def sample_window(model, batch, prompt, visible, cfg, device, generator, predictor):
    B, T, _ = prompt.shape
    steps = cfg["FLOW_STEPS"]
    noise = torch.randn((B, T, D), device=device, generator=generator)
    state = noise.clone()
    with autocast_context(cfg, device):
        cond = model.condition(batch)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        pred = predictor(state, t, batch, prompt, visible, cond, phase="causal")
        clean = project_state(pred.float())
        te = (i + 1) / steps
        state = noise * (1 - te) + clean * te
    if not torch.isfinite(state).all():
        raise FloatingPointError("Nonfinite generated state")
    return state


@torch.inference_mode()
def generate(model, sample, stats, cfg, device, seed=None, max_frames=None, predictor=None, should_stop=None):
    model.eval()
    if predictor is None:
        with CFGPredictor(model, cfg, device) as engine:
            result = generate(model, sample, stats, cfg, device, seed, max_frames, engine, should_stop)
            result["inference"] = engine.report()
            return result
    L = int(cfg["WINDOW_FRAMES"])
    H = int(cfg["HISTORY_FRAMES"])
    hop = L - H
    n = 2 * len(sample["hidden"])
    n = min(n, max_frames) if max_frames is not None else n
    if n <= 0:
        raise ValueError("Empty generation request")
    rng = torch.Generator(device=device).manual_seed(
        stable_seed(sample["uid"], cfg["EVAL_SEED"]) if seed is None else int(seed)
    )
    previous = neutral_history(H, device)
    outputs = []
    boundaries = []
    started = time.perf_counter()
    for cursor in range(0, n, hop):
        if should_stop is not None and should_stop():
            raise GenerationStopped("Stop requested during generation")
        indices = np.arange((cursor - H) // 2, (cursor - H + L) // 2).clip(0, len(sample["hidden"]) - 1)
        batch = {"pid": torch.tensor([sample["pid"]], device=device)}
        batch["hidden"] = torch.from_numpy(sample["hidden"][indices])[None].to(device)
        offset = previous[0, 330:333].clone()
        prompt = torch.zeros((1, L, D), device=device)
        prompt[0, :H] = previous
        prompt[0, :H, 330:333] -= offset
        visible = torch.zeros_like(prompt, dtype=torch.bool)
        visible[:, :H] = True
        state = sample_window(model, batch, prompt, visible, cfg, device, rng, predictor)
        fresh = state[0, H:].clone()
        fresh[:, 330:333] += offset
        take = min(hop, n - cursor)
        outputs.append(fresh[:take].cpu().numpy())
        previous = torch.cat([previous, fresh[:take]], 0)[-H:]
        if cursor:
            boundaries.append(cursor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    state = np.concatenate(outputs, 0)
    expression = state[:, 333:] * stats["expr_std"] + stats["expr_mean"]
    return {
        "state": state,
        "poses": state_to_local_aa(state),
        "trans": state[:, 330:333],
        "expressions": expression,
        "mocap_frame_rate": 25,
        "window_seams25": np.asarray(boundaries, np.int64),
        "cache_seams12p5": sample["seams"],
        "generation_seconds": time.perf_counter() - started,
    }
