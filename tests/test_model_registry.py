# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Smoke tests for the model registry and the QAOAModel forward pass.

Every architecture must construct and produce a correctly shaped prediction when
driven through ``QAOAModel``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ml_models.models import MODEL_REGISTRY, get_model
from training.qaoa_model import QAOAModel

IN_FEATURES = [
    "num_nodes",
    "num_edges",
    "edges_per_node",
    "mean_degree",
    "std_degree",
]
INPUT_DIM = len(IN_FEATURES)
P = 1
OUTPUT_DIM = 2 * P
NUM_NODES = 8
BATCH_SIZE = 3

ALL_MODEL_NAMES = sorted(MODEL_REGISTRY)

def make_batch(with_edge_weights: bool = False) -> dict:
    """Build a batch shaped like the one the datamodule's collate produces.

    Scalar features are per-graph tensors; ``edges`` is a list of (M, 2) tensors,
    one per graph, rather than a single batched edge_index.
    """
    ring = torch.tensor(
        [[i, (i + 1) % NUM_NODES] for i in range(NUM_NODES)], dtype=torch.long
    )

    batch = {name: torch.rand(BATCH_SIZE) for name in IN_FEATURES}
    batch["node_count"] = torch.full((BATCH_SIZE,), NUM_NODES, dtype=torch.long)
    batch["edges"] = [ring.clone() for _ in range(BATCH_SIZE)]
    # Diffusion models consume a timestep per graph.
    batch["t"] = torch.zeros(BATCH_SIZE, dtype=torch.long)

    if with_edge_weights:
        batch["edge_weights"] = [
            torch.ones(ring.shape[0], dtype=torch.float32) for _ in range(BATCH_SIZE)
        ]

    return batch


def make_model(model_type: str) -> QAOAModel:
    # diffusion_transformer requires a positive timestep count; the value is
    # ignored by every other architecture.
    timesteps = 100 if "diffusion" in model_type else 0
    return QAOAModel(
        model_type=model_type,
        in_features=IN_FEATURES,
        target_key="qaoa_params",
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        lr=1e-3,
        depth=P,
        timesteps=timesteps,
    )


@pytest.mark.parametrize("model_type", ALL_MODEL_NAMES)
def test_model_constructs(model_type):
    """Every registry key and alias builds a module with trainable parameters."""
    model = get_model(
        model_type,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        node_feature_dim=0,
    )
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


@pytest.mark.parametrize("model_type", ALL_MODEL_NAMES)
@pytest.mark.parametrize("with_edge_weights", [False, True], ids=["plain", "weighted"])
def test_forward_pass_output_shape(model_type, with_edge_weights):
    """A forward pass returns one angle vector of length 2p per graph."""
    model = make_model(model_type)
    model.eval()

    with torch.no_grad():
        out = model(make_batch(with_edge_weights=with_edge_weights))

    assert out.shape == (BATCH_SIZE, OUTPUT_DIM), (
        f"{model_type}: expected {(BATCH_SIZE, OUTPUT_DIM)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"{model_type}: non-finite outputs"


@pytest.mark.parametrize("model_type", ALL_MODEL_NAMES)
def test_backward_pass_produces_gradients(model_type):
    """Loss on the prediction must reach the model parameters."""
    model = make_model(model_type)
    model.train()

    out = model(make_batch(with_edge_weights=False))
    out.sum().backward()

    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, f"{model_type}: no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads), (
        f"{model_type}: all gradients are zero"
    )


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unsupported model type"):
        get_model("no_such_model", input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)



def test_model_type_is_case_insensitive():
    upper = get_model("GRAPH_NEURAL_NETWORK", input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    lower = get_model("graph_neural_network", input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    assert type(upper) is type(lower)
