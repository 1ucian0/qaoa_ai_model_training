# Copyright IBM Corp. 2026
# License: Apache-2.0

import torch.nn as nn

EMBED_DIM = 64
NUM_LAYERS = 4
NUM_HEADS = 4
POS_ENC_DIM = 8
EDGE_DIM = 1


def _mlp(input_dim, output_dim, **_):
    from ml_models.mlp import MLP

    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
    )


def _edge_transformer(input_dim, output_dim, **_):
    from ml_models.edge_transformer import EdgeTransformer

    return EdgeTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        n_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        use_positional_encoding=True,
        permute_nodes_during_training=True,
    )


def _diffusion_transformer(input_dim, output_dim, timesteps=100, **_):
    from ml_models.ddpm_transformer import DDPMTransformer

    return DDPMTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        n_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        timesteps=timesteps,
    )


def _graph_neural_network(input_dim, output_dim, node_feature_dim=0, **_):
    from ml_models.graph_neural_network import GNNRegression

    return GNNRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        edge_dim=EDGE_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        node_feature_dim=node_feature_dim,
    )


def _graph_isomorphism_network(input_dim, output_dim, node_feature_dim=0, **_):
    from ml_models.graph_isomorphism_network import GINRegression

    return GINRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        edge_dim=EDGE_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        node_feature_dim=node_feature_dim,
    )


def _graph_transformer(input_dim, output_dim, node_feature_dim=0, **_):
    from ml_models.graph_transformer import GraphTransformer

    return GraphTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=EMBED_DIM * 2,
        pos_enc_dim=POS_ENC_DIM,
        node_feature_dim=node_feature_dim,
        random_sign_flip_pe=True,
    )


def _gcn(input_dim, output_dim, **_):
    from ml_models.graph_convolutional_network import GCNModel

    return GCNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=EMBED_DIM,
    )


MODEL_REGISTRY = {
    "mlp": _mlp,
    "edge_transformer": _edge_transformer,
    "diffusion_transformer": _diffusion_transformer,
    "graph_neural_network": _graph_neural_network,
    "graph_isomorphism_network": _graph_isomorphism_network,
    "graph_transformer": _graph_transformer,
    "gcn": _gcn,
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model type {model_type!r}. "
            f"Available models: {sorted(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_type](**kwargs)
