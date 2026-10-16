import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from .geometry import D, part_indices


class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_persons, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t):
        return self.mlp(t)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size % 3 != 0:
            raise ValueError("Parameter-matched SwiGLU requires hidden_size divisible by 3")
        intermediate_size = hidden_size * 4 // 3
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 9))
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = SwiGLU(hidden_size)
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)
        self.ln_poses_prompt = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_poses_prompt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, x, poses_prompt, time_embed, prompt_mask):
        time_emb = time_embed
        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(time_emb).chunk(
            9, dim=-1
        )
        poses_prompt0 = poses_prompt
        poses_prompt = self.ln_poses_prompt(poses_prompt)
        poses_prompt = poses_prompt * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        poses_prompt = self.ff_poses_prompt(poses_prompt) * gate3.unsqueeze(1) + poses_prompt0
        x = (
            self.cross_attn(x, poses_prompt, poses_prompt, key_padding_mask=prompt_mask, need_weights=False)[
                0
            ]
            + x
        )
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn(x, x, x, need_weights=False)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x


class Fusion(nn.Module):
    def __init__(self, feature_size=384, hidden_size=512):
        super().__init__()
        self.in_prj = nn.Linear(feature_size, hidden_size)
        self.out_prj = nn.Linear(hidden_size, feature_size)
        self.atten = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, *parts):
        original = torch.cat(parts, dim=1)
        x = self.in_prj(original)
        x = self.out_prj(self.atten(x, x, x, need_weights=False)[0]) + original
        return x.chunk(4, dim=1)


class MotionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size, num_layers, num_person, expression_dim = (
            cfg["MODEL_WIDTH"],
            cfg["MODEL_LAYERS"],
            cfg["NUM_SPEAKERS"],
            100,
        )
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear(132, hidden_size)
        self.in_trans = nn.Linear(3, hidden_size)
        self.in_x_hand = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)
        self.in_x_trans = nn.Linear(55 * 6 + 3 + expression_dim, hidden_size)
        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.trans_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.transformer_hand = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_trans = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.fusion_blocks = nn.ModuleList([Fusion(feature_size=hidden_size) for _ in range(num_layers - 1)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, 132)
        self.trans = nn.Linear(hidden_size, 3)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size - 64)
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person
        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear(132, hidden_size)
        self.seed_prj_trans = nn.Linear(3, hidden_size)
        self.voxtream_prj = nn.Sequential(
            nn.Linear(1024, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.in_face = nn.Linear(118, hidden_size)
        self.in_x_face = nn.Linear(D, hidden_size)
        self.face_fuse = nn.Linear(3 * hidden_size, hidden_size)
        self.seed_prj_face = nn.Linear(118, hidden_size)
        self.face = nn.Linear(hidden_size, 118)
        self.transformer_face = nn.ModuleList([DiTBlock(hidden_size) for _ in range(num_layers)])
        self.cache_adapter = nn.Linear(cfg["TT_DIM"], 1024)
        if cfg["TT_DIM"] == 1024:
            nn.init.eye_(self.cache_adapter.weight)
            nn.init.zeros_(self.cache_adapter.bias)
        self.config = cfg.copy()
        self.part_names = ("hand", "body", "trans", "face")
        parts, _ = part_indices()
        for name, indices in parts.items():
            self.register_buffer("indices_" + name, torch.as_tensor(indices), persistent=False)

    def condition(self, batch, drop=None):
        condition = self.cache_adapter(batch["hidden"].float())
        if drop is not None:
            condition = condition.masked_fill(drop[:, None, None], 0)
        return condition.repeat_interleave(2, dim=1)

    def forward(self, x, time_steps, batch, prompt, visible, drop=None, condition=None):
        if condition is None:
            condition = self.condition(batch, drop)
        speech = self.voxtream_prj(condition)
        pid = batch["pid"].long()
        ts = torch.cat(
            [
                self.time_embed(time_steps.reshape(-1, 1)),
                self.style_embed(F.one_hot(pid, num_classes=self.num_person).float()),
            ],
            -1,
        )
        states = []
        prompts = []
        masks = []
        for name in self.part_names:
            idx = getattr(self, "indices_" + name)
            own = getattr(self, "in_" + name)(x[..., idx])
            glob = getattr(self, "in_x_" + name)(x)
            pieces = [own, glob] if name == "trans" else [own, glob, speech]
            states.append(getattr(self, name + "_fuse")(torch.cat(pieces, -1)))
            vis = visible[..., idx]
            pr = prompt[..., idx].masked_fill(~vis, 0)
            mask = ~vis.any(-1)
            all_hidden = mask.all(-1)
            mask = mask.clone()
            mask[:, 0] = mask[:, 0] & ~all_hidden
            p = self.pos_enc(getattr(self, "seed_prj_" + name)(pr))
            prompts.append(p)
            masks.append(mask)
        states = list(self.pos_enc(torch.cat(states, 1)).chunk(4, 1))
        for layer in range(self.config["MODEL_LAYERS"]):
            for k, name in enumerate(self.part_names):
                block = getattr(self, "transformer_" + name)[layer]

                def apply_block(s, p, t, m, block=block):
                    return block(s, poses_prompt=p, time_embed=t, prompt_mask=m)

                if self.training and self.config["GRADIENT_CHECKPOINTING"]:
                    states[k] = checkpoint(
                        apply_block, states[k], prompts[k], ts, masks[k], use_reentrant=False
                    )
                else:
                    states[k] = apply_block(states[k], prompts[k], ts, masks[k])
            if layer < len(self.fusion_blocks):
                fusion = self.fusion_blocks[layer]
                if self.training and self.config["GRADIENT_CHECKPOINTING"]:
                    states = list(checkpoint(fusion, *states, use_reentrant=False))
                else:
                    states = list(fusion(*states))
        output = x.new_zeros(x.shape, dtype=torch.float32)
        for name, s in zip(self.part_names, states):
            branch = getattr(self, name)(s).float()
            output[..., getattr(self, "indices_" + name)] = branch
        return output
