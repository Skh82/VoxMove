import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

D = 433
PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    15,
    15,
    15,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    20,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
    21,
    52,
    53,
]
JOINT_GROUPS = {
    "hand": [
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
    ],
    "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "lower": [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "face": [22, 23, 24],
}


def part_indices(cfg=None):
    groups = {
        name: (np.asarray(joints)[:, None] * 6 + np.arange(6)).reshape(-1)
        for name, joints in JOINT_GROUPS.items()
    }
    parts = {
        "hand": groups["hand"],
        "body": np.r_[groups["upper"], groups["lower"]],
        "trans": np.arange(330, 333),
        "face": np.r_[groups["face"], np.arange(333, 433)],
    }
    return (parts, groups)


def rot6_to_matrix(x):
    a, b = (x.float()[..., :3], x.float()[..., 3:6])
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    a = torch.where(a.norm(dim=-1, keepdim=True) > 1e-06, a, fallback)
    u = F.normalize(a, dim=-1)
    v = b - (u * b).sum(-1, keepdim=True) * u
    alt = F.one_hot(u.abs().argmin(-1), 3).to(u)
    alt = alt - (alt * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(v.norm(dim=-1, keepdim=True) > 1e-06, v, alt), dim=-1)
    return torch.stack([u, v, torch.cross(u, v, dim=-1)], dim=-2)


def project_state(x):
    r = rot6_to_matrix(x[..., :330].reshape(*x.shape[:-1], 55, 6))[..., :2, :].flatten(-3)
    return torch.cat([r, x[..., 330:].float()], -1)


def local_to_global(local):
    result = [local[..., 0, :, :]]
    for j in range(1, 55):
        result.append(result[PARENTS[j]] @ local[..., j, :, :])
    return np.stack(result, axis=-3)


def state_to_local_aa(x):
    g = rot6_to_matrix(torch.as_tensor(x[..., :330]).reshape(-1, 55, 6)).numpy()
    local = g.copy()
    for j in range(1, 55):
        local[:, j] = g[:, PARENTS[j]].transpose(0, 2, 1) @ g[:, j]
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 55, 3).astype(np.float32)


def fk_global(rot, rest, virtual=False, include_root=False):
    pos = [rest[..., 0, :] if include_root else torch.zeros_like(rest[..., 0, :])]
    for j in range(1, 55):
        bone = rest[..., j, :] - rest[..., PARENTS[j], :]
        pos.append(pos[PARENTS[j]] + (rot[..., PARENTS[j], :, :] @ bone.unsqueeze(-1)).squeeze(-1))
    pos = torch.stack(pos, -2)
    if virtual:
        offsets = rot.new_tensor(
            [[-0.05, 0, 0], [0.05, 0, 0], [0, -0.05, 0], [0, 0.05, 0], [0, 0, -0.05], [0, 0, 0.05]]
        )
        extra = torch.einsum("...jik,vk->...jvi", rot, offsets) + pos.unsqueeze(-2)
        pos = torch.cat([pos, extra.flatten(-3, -2)], -2)
    return pos
