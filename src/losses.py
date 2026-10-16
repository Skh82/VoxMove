import math
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import PARENTS, part_indices, rot6_to_matrix, fk_global


def causal_visible(x, valid, cfg):
    B, T, _ = x.shape
    H = int(cfg["HISTORY_FRAMES"])
    history = torch.arange(T, device=x.device)[None, :] < H
    visible = history[..., None].expand_as(x)
    return visible & (valid > 0)


def masked_mean(x, mask):
    mask = torch.broadcast_to(mask, x.shape).to(x)
    return (x * mask).sum() / mask.sum().clamp_min(1)


class TemporalContrastiveLoss:
    VERSION = 1
    LENGTHS = (8, 16, 32)
    TEMPERATURE = 0.2

    def __init__(self, cfg, state=None):
        parts, _ = part_indices(cfg)
        self.groups = {name: np.unique(parts[name] // 6).tolist() for name in ("hand", "body")}
        self.groups["body"] = [j for j in self.groups["body"] if j != 0]
        self.stats = {}
        self.device_stats = {}
        if state is not None:
            if state["version"] != self.VERSION or tuple(state["lengths"]) != self.LENGTHS:
                raise ValueError("Temporal-loss checkpoint recipe mismatch")
            if state["temperature"] != self.TEMPERATURE or state["groups"] != self.groups:
                raise ValueError("Temporal-loss checkpoint feature definition mismatch")
            self.stats = state["stats"]

    def state_dict(self):
        return {
            "version": self.VERSION,
            "lengths": list(self.LENGTHS),
            "temperature": self.TEMPERATURE,
            "groups": self.groups,
            "normalization": "frozen_first_valid_global_training_minibatch",
            "stats": self.stats,
        }

    @staticmethod
    def describe(position):
        n, length = position.shape[:2]
        position = position.float().flatten(2)
        velocity = 25.0 * torch.diff(position, dim=1)
        hann = torch.hann_window(length - 1, periodic=False, device=position.device, dtype=torch.float32)
        spectrum = torch.log1p(
            torch.fft.rfft(velocity * hann[None, :, None], n=length, dim=1, norm="ortho").abs()
        )
        blocks = [position.flatten(1), velocity.flatten(1), spectrum.flatten(1)]
        return (torch.cat(blocks, dim=-1), [b.shape[-1] for b in blocks])

    @torch.no_grad()
    def calibrate(self, real, valid):
        import torch.distributed as dist

        for name, ids in self.groups.items():
            position = real[:, :, ids]
            good = valid[:, :, ids].all(-1)
            for length in self.LENGTHS:
                key = f"{name}_{length}"
                if key in self.stats:
                    continue
                windows = position.unfold(1, length, length).permute(0, 1, 4, 2, 3)
                windows = windows.reshape(-1, length, len(ids), 3)
                keep = good.unfold(1, length, length).all(-1).flatten()
                values, widths = self.describe(windows)
                values = values[keep].double()
                moments = torch.cat(
                    [values.sum(0), values.square().sum(0), values.new_tensor([values.shape[0]])]
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(moments, op=dist.ReduceOp.SUM)
                count = int(moments[-1].item())
                if count < 2:
                    continue
                width = sum(widths)
                mean = moments[:width] / count
                variance = (moments[width : 2 * width] / count - mean.square()).clamp_min(0)
                floors = torch.cat(
                    [values.new_full((d,), floor) for d, floor in zip(widths, (0.01, 0.1, 0.05))]
                )
                std = torch.maximum(variance.sqrt(), floors)
                balance = torch.cat([values.new_full((d,), 1 / math.sqrt(3 * d)) for d in widths])
                self.stats[key] = {
                    "mean": mean.float().cpu(),
                    "std": std.float().cpu(),
                    "balance": balance.float().cpu(),
                    "windows": count,
                }
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[TEMPORAL_STATS] {key}: frozen from {count} valid training windows", flush=True)

    def normalize(self, position, key):
        cache_key = (key, str(position.device))
        if cache_key not in self.device_stats:
            self.device_stats[cache_key] = {
                k: self.stats[key][k].to(position.device) for k in ("mean", "std", "balance")
            }
        stats = self.device_stats[cache_key]
        values, _ = self.describe(position)
        return (values - stats["mean"]) / stats["std"] * stats["balance"]

    def logits(self, query, keys):
        distance = (
            query.square().sum(-1, keepdim=True) + keys.square().sum(-1)[None] - 2 * query @ keys.T
        ).clamp_min(0)
        return -distance / self.TEMPERATURE

    @torch.no_grad()
    def comparison_keys(self, real, key):
        mean = real.mean(1, keepdim=True)
        held = mean.expand_as(real)
        weakened = mean + 0.25 * (real - mean)
        clock = torch.arange(real.shape[1], device=real.device, dtype=torch.float32) / 25.0
        phase = torch.arange(real.shape[2] * 3, device=real.device, dtype=torch.float32)
        phase = phase.reshape(1, 1, real.shape[2], 3) * 0.618
        jitter = 0.01 * torch.sin(2 * math.pi * 10 * clock[None, :, None, None] + phase)
        keys = self.normalize(torch.cat([real, held, weakened, real + jitter], dim=0), key)
        log_target = self.logits(keys[: len(real)], keys).log_softmax(-1)
        return (keys, log_target)

    def __call__(self, pred, real, valid, visible, time_steps):
        pred = pred[:, :, :55].float() - pred[:, :, :1].float()
        real = real[:, :, :55].detach().float() - real[:, :, :1].detach().float()
        valid = valid.bool()
        self.calibrate(real, valid)
        terms = []
        active_windows = 0
        for name, ids in self.groups.items():
            query_all, real_all = (pred[:, :, ids], real[:, :, ids])
            eligible = valid[:, :, ids].all(-1) & ~visible[:, :, ids].any(-1)
            eligible = eligible & (time_steps[:, None] >= 0.5)
            for length in self.LENGTHS:
                key = f"{name}_{length}"
                if key not in self.stats:
                    continue
                choices = eligible.unfold(1, length, 1).all(-1)
                rows = choices.any(-1).nonzero(as_tuple=True)[0]
                if len(rows) == 0:
                    continue
                starts = choices[rows].long().argmax(-1)
                frames = starts[:, None] + torch.arange(length, device=pred.device)[None]
                query_position = query_all[rows[:, None], frames]
                real_position = real_all[rows[:, None], frames]
                keys, log_target = self.comparison_keys(real_position, key)
                log_query = self.logits(self.normalize(query_position, key), keys).log_softmax(-1)
                kl = (log_target.exp() * (log_target - log_query)).sum(-1).clamp_min(0)
                terms.append(kl.mean())
                active_windows += len(rows)
        loss = torch.stack(terms).mean() if terms else pred.new_zeros(())
        return (loss, pred.new_tensor(float(active_windows)))


def temporal_loss_state(model):
    owner = model.module if hasattr(model, "module") else model
    helper = getattr(owner, "_temporal_loss", None)
    return helper.state_dict() if helper is not None else getattr(owner, "_temporal_loss_state", None)


def losses(pred, target, valid, prompt, visible, rest, cfg, temporal_context=None):
    pred, target = (pred.float(), target.float())
    B, T, _ = pred.shape
    r = rot6_to_matrix(pred[..., :330].reshape(B, T, 55, 6))
    gt = rot6_to_matrix(target[..., :330].reshape(B, T, 55, 6))
    jvalid = valid[..., :330].reshape(B, T, 55, 6).amin(-1)
    relative = gt @ r.transpose(-1, -2)
    eye = torch.eye(3, device=pred.device)
    recon = masked_mean((relative - eye).abs(), jvalid[..., None, None])
    recon += masked_mean((pred[..., 330:333] - target[..., 330:333]).abs(), valid[..., 330:333])
    chain = [jvalid[..., 0]]
    for j in range(1, 55):
        chain.append(jvalid[..., j] * chain[PARENTS[j]])
    chain = torch.stack(chain, -1)
    posmask = torch.cat([chain, chain.repeat_interleave(6, -1)], -1)[..., None]
    joints = rest[:, None].expand(-1, T, -1, -1).float()
    p = fk_global(r, joints, True) + pred[..., None, 330:333]
    q = fk_global(gt, joints, True) + target[..., None, 330:333]
    posmask = posmask * valid[..., 330:333].amin(-1)[..., None, None]
    geometric = masked_mean(F.smooth_l1_loss(p, q, reduction="none", beta=0.02), posmask)
    for order in [1, 2]:
        pv = torch.diff(p, n=order, dim=1) * (25 / 30) ** order
        qv = torch.diff(q, n=order, dim=1) * (25 / 30) ** order
        vm = posmask[:, : T - order].clone()
        for shift in range(1, order + 1):
            vm = vm * posmask[:, shift : shift + T - order]
        geometric += 10 * masked_mean(F.smooth_l1_loss(pv, qv, reduction="none", beta=0.02), vm)
    parent = torch.as_tensor(PARENTS[1:], device=pred.device)
    a = F.normalize(p[..., :55, :][..., 1:, :] - p[..., :55, :][..., parent, :], dim=-1)
    b = F.normalize(q[..., :55, :][..., 1:, :] - q[..., :55, :][..., parent, :], dim=-1)
    pair = chain[..., 1:, None] * chain[..., None, 1:]
    structure = masked_mean((a @ a.transpose(-1, -2) - b @ b.transpose(-1, -2)).abs(), pair)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    angular = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-06, 1 - 1e-06))
    pm = visible[..., :330].reshape(B, T, 55, 6).all(-1)
    prompt_rot = masked_mean(angular, pm)
    prompt_trans = masked_mean((pred[..., 330:333] - prompt[..., 330:333]).abs(), visible[..., 330:333])
    face = masked_mean((pred[..., 333:] - target[..., 333:]) ** 2, valid[..., 333:])
    total = recon + cfg["GEOMETRIC_LOSS_WEIGHT"] * geometric + cfg["STRUCTURE_LOSS_WEIGHT"] * structure
    total += 0.1 * (prompt_rot + prompt_trans) + cfg["EXPRESSION_LOSS_WEIGHT"] * face
    terms = {
        "total": total,
        "reconstruction": recon,
        "geometric": geometric,
        "structure": structure,
        "expression": face,
    }
    if temporal_context is not None:
        helper, time_steps = temporal_context
        temporal, active = helper(
            p, q, chain > 0, visible[..., :330].reshape(B, T, 55, 6).any(-1), time_steps
        )
        weighted = float(cfg["TEMPORAL_CONTRASTIVE_WEIGHT"]) * temporal
        terms.update(
            base_total=total,
            temporal_loss=temporal,
            temporal_loss_weighted=weighted,
            temporal_active_windows=active,
        )
        terms["total"] = total + weighted
    return terms


def flow_loss(model, batch, cfg, training=True):
    x = batch["x"]
    visible = causal_visible(x, batch["valid"], cfg)
    H = int(cfg["HISTORY_FRAMES"])
    prompt = torch.zeros_like(x)
    prompt[:, :H] = x[:, :H]
    t = torch.rand(len(x), device=x.device)
    noise = torch.randn_like(x)
    xt = noise * (1 - t[:, None, None]) + x * t[:, None, None]
    drop = torch.rand(len(x), device=x.device) < cfg["CFG_DROPOUT"] if training else None
    pred = model(xt, t, batch, prompt, visible, drop=drop)
    with torch.autocast(device_type=x.device.type, enabled=False):
        temporal_context = None
        if training and float(cfg.get("TEMPORAL_CONTRASTIVE_WEIGHT", 0.0)) > 0:
            owner = model.module if hasattr(model, "module") else model
            if not hasattr(owner, "_temporal_loss"):
                owner._temporal_loss = TemporalContrastiveLoss(
                    cfg, getattr(owner, "_temporal_loss_state", None)
                )
            temporal_context = (owner._temporal_loss, t)
        terms = losses(pred, x, batch["valid"], prompt, visible, batch["rest"], cfg, temporal_context)
        if training and temporal_context is None:
            zero = terms["total"].new_zeros(())
            terms.update(
                base_total=terms["total"],
                temporal_loss=zero,
                temporal_loss_weighted=zero,
                temporal_active_windows=zero,
            )
    return terms
