# Copyright IBM Corp. 2026
# License: Apache-2.0

"""
Custom collate function for batching graphs with variable-sized node and edge features.

Handles:
- Scalar features: Stack into (B,)
- Node features: List of tensors (different size per graph)
- Edges / edge weights / edge features: List of tensors (different size per graph)
"""

import torch
from typing import List, Dict, Tuple
from datamodule.meta_configs import NODE_FEATURES, EDGE_FEATURES

def qaoa_collate_fn(batch: List[Tuple[Dict, Dict]]) -> Tuple[Dict, Dict]:
    """
    Custom collate function for QAOA dataset with variable-sized node/edge features.

    Args:
        batch: List of (x_dict, y_dict) tuples from dataset

    Returns:
        Tuple of (batched_x_dict, batched_y_dict)
    """
    if len(batch) == 0:
        return {}, {}

    # Separate x and y
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    # Get all keys
    x_keys = set()
    y_keys = set()
    for x in x_list:
        x_keys.update(x.keys())
    for y in y_list:
        y_keys.update(y.keys())

    # Collate x (inputs)
    batched_x = {}
    for key in x_keys:
        values = [x[key] for x in x_list if key in x]

        if len(values) == 0:
            continue

        # Variable-length per-graph tensors: keep as list, models iterate per graph.
        if key in set(NODE_FEATURES) | set(EDGE_FEATURES) | {"edges", "edge_weights"}:
            batched_x[key] = values

        elif all(isinstance(v, torch.Tensor) for v in values):
            v0 = values[0]
            # Scalars (0-d or single-element): reshape to 0-d then stack into (B,).
            if v0.dim() == 0 or v0.numel() == 1:
                batched_x[key] = torch.stack([v.reshape(()) for v in values], dim=0)
            else:
                batched_x[key] = torch.stack(values, dim=0)
        else:
            # Non-tensor or mixed (int, float, str, list, etc.) - keep as list
            batched_x[key] = values

    # Collate y (outputs)
    batched_y = {}
    for key in y_keys:
        values = [y[key] for y in y_list if key in y]

        if len(values) == 0:
            continue

        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                batched_y[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # If stacking fails, keep as list
                batched_y[key] = values
        else:
            # Non-tensor - keep as list
            batched_y[key] = values

    return batched_x, batched_y