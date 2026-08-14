# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch to avoid segfault warning
# See: https://github.com/pytorch/pytorch/issues/78829
try:
    import juliacall
except ImportError:
    pass  # juliacall is optional

import re
import torch
from collections import defaultdict
import numpy as np
import torch
import random


def set_seed(seed: int) -> None:
    """Seed random, numpy, torch, and Lightning together."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

def normalize_qaoa_params(parameters: torch.Tensor, scale: float = torch.pi / 2) -> torch.Tensor:
    """
    Normalize QAOA parameters stored as [beta_1,...beta_p, gamma_1,...gamma_p].

    Beta and gamma are linearly scaled to [0, 1].
    """
    return parameters / scale

def denormalize_qaoa_params(qaoa_params_norm: torch.Tensor, scale: float = torch.pi / 2) -> torch.Tensor:
    """
    Inverse of normalize_qaoa_params.
    """
    return qaoa_params_norm * scale

def rescaling_factor(cost_op):
    terms = defaultdict(list)

    for p in cost_op:
        order = sum(p.paulis[0].z)
        terms[order].append(np.real(p.coeffs[0]) ** 2)

    factor = 0
    for squared_weights in terms.values():
        factor += sum(squared_weights) / len(squared_weights)

    return np.sqrt(factor)

def angles_to_target(angles: list[float], p: int, rescale_a: float) -> torch.Tensor:
    """
    Check again if p and len(angles) matches and return torch tensor.
    Applies rescaling to the second half of angles (gammas).
    """
    p = int(p)
    if len(angles) != 2 * p:
        raise ValueError(f"qaoa_angles must have length 2*p. Got {len(angles)} for p={p}.")

    angles = list(angles)  # ensure mutable

    # rescale second half by gamma rescale factor
    angles[p:] = [a * rescale_a for a in angles[p:]]

    return torch.tensor(angles, dtype=torch.float32)


def undo_gamma_rescale(angles: torch.Tensor, p: int, rescale_a: torch.Tensor | float) -> torch.Tensor:
    angles = angles.clone()

    if not torch.is_tensor(rescale_a):
        rescale_a = torch.tensor(rescale_a, dtype=angles.dtype, device=angles.device)
    else:
        rescale_a = rescale_a.to(dtype=angles.dtype, device=angles.device)

    if rescale_a.ndim == 1:
        rescale_a = rescale_a.unsqueeze(-1)

    angles[..., p:] = angles[..., p:] / rescale_a
    return angles