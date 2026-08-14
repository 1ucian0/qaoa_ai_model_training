# Copyright IBM Corp. 2026
# License: Apache-2.0

import lightning.pytorch as pl
from torch import nn
import torch
from pathlib import Path
import math
import warnings

from datamodule.meta_configs import NODE_FEATURES, EDGE_FEATURES
from ml_models.models import get_model
from .eval_step import QAOAEvalMixin

from torch_geometric.data import Data, Batch


def _symmetrize_edges(
    edges_mx2: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add the reverse of every edge so message passing sees an undirected graph.

    Args:
        edges_mx2:   Edge list of shape (M, 2) = [src, dst] per row.
        edge_weight: Optional per-edge weights, shape (M,) or (M, 1).

    Returns:
        (edges_2mx2, weight_2m), edges of shape (2M, 2) with each edge
        followed by its reverse, and weights duplicated to match (or None).
    """
    if edges_mx2.numel() == 0:
        return edges_mx2, edge_weight
    reversed_edges = edges_mx2.flip(1)  # (M, 2): swap [src, dst] -> [dst, src]
    sym_edges = torch.cat([edges_mx2, reversed_edges], dim=0)  # (2M, 2)
    sym_weight = (
        torch.cat([edge_weight, edge_weight], dim=0)
        if edge_weight is not None
        else None
    )
    return sym_edges, sym_weight

class QAOAModel(QAOAEvalMixin, pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        in_features: list[str],
        target_key: str,
        input_dim: int,
        output_dim: int,
        lr: float,
        depth: int,
        instance_dir: str = None,
        fixed_angle_dir: str = None,
        timesteps: int = 0,
        use_degree_flag: bool = False,
        use_triangle_features: bool = False,
        summary_file: str = None,
        test_eval_workers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_features = list(in_features)
        self.target_key = str(target_key)
        self.depth = int(depth)
        self.instance_dir = Path(instance_dir) if instance_dir else None
        self.fixed_angle_dir = Path(fixed_angle_dir) if fixed_angle_dir else None
        self.test_eval_workers = int(test_eval_workers)

        # Width of the per-node feature vector the graph models will receive
        node_feats = [f for f in self.in_features if f in NODE_FEATURES]
        node_feature_dim = sum(
            3 if f == "node_triangle_features_combined" else 1 for f in node_feats
        )

        self.core = get_model(
            model_type,
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            timesteps=int(timesteps),
            use_degree_flag=bool(use_degree_flag),
            use_triangle_features=bool(use_triangle_features),
            node_feature_dim=int(node_feature_dim),
        )
        self.criterion = nn.MSELoss()

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

    def unpack_x(self, x: dict[str, any]) -> torch.Tensor:
        """
        Extract and concatenate scalar/graph-level features only.
        Node and edge features are handled separately by the model.
        """

        skip_features = set(NODE_FEATURES) | set(EDGE_FEATURES) | {"edges", "edge_weights"}
        parts: list[torch.Tensor] = []

        # Work around for no-scalar ablations
        if "node_count" in x and torch.is_tensor(x["node_count"]):
            batch_size = int(x["node_count"].numel())
        elif "edges" in x and isinstance(x["edges"], (list, tuple)):
            batch_size = len(x["edges"])
        else:
            batch_size = 1

        for f in sorted(self.in_features):
            if f in skip_features:
                continue

            v = x.get(f, torch.zeros(batch_size))

            if torch.is_tensor(v):
                v = v.detach().to(dtype=torch.float32, device=self.device)
            else:
                v = torch.as_tensor(v, dtype=torch.float32, device=self.device)

            if v.dim() == 0:
                v = v.view(1, 1)
            elif v.dim() == 1:
                v = v.unsqueeze(1)
            elif v.dim() != 2:
                raise ValueError(f"Unexpected tensor dimension for feature {f}: {v.shape}")

            parts.append(v)

        if not parts:
            return torch.empty(batch_size, 0, device=self.device)

        return torch.cat(parts, dim=1)


    def forward(self, x: dict[str, any] | torch.Tensor) -> torch.Tensor:
        model_type = str(self.hparams.model_type).lower()
        x_vec = self.unpack_x(x)

        if model_type == "mlp":
            return self.core(x_vec)

        if model_type == "edge_transformer":
            if "edges" not in x:
                raise KeyError("edge_transformer requires 'edges' in input")
            edges = [e.to(self.device) for e in x["edges"]]

            if "edge_weights" in x:
                edge_weights = [w.to(self.device) for w in x["edge_weights"]]
                return self.core(x_vec, edges, edge_weights)
            return self.core(x_vec, edges)

        if model_type == "diffusion_transformer":
            if "edges" not in x:
                raise KeyError("diffusion_transformer requires 'edges' in input")
            if "t" not in x:
                raise KeyError("diffusion_transformer requires 't' in input")

            edges = [e.to(self.device) for e in x["edges"]]
            t = x["t"].to(self.device)

            if "edge_weights" in x:
                edge_weights = [w.to(self.device) for w in x["edge_weights"]]
                return self.core(x_vec, edges, t, edge_weights)
            return self.core(x_vec, edges, t)

        if model_type == "graph_transformer":
            if "edges" not in x:
                raise KeyError("graph_transformer requires 'edges' in input")
            if "node_count" not in x:
                raise KeyError("graph_transformer requires 'node_count' in input")

            batch_size = x_vec.shape[0]
            raw_edges = [e.to(self.device) for e in x["edges"]]
            raw_weights = (
                [w.to(self.device) for w in x["edge_weights"]]
                if "edge_weights" in x
                else [None] * len(raw_edges)
            )
            node_count = x["node_count"].to(self.device)
            # Symmetrize edges
            sym = [_symmetrize_edges(e, w) for e, w in zip(raw_edges, raw_weights)]
            edges = [e for e, _ in sym]
            edge_weights = None if "edge_weights" not in x else [w for _, w in sym]

            requested_node_features = [feat for feat in NODE_FEATURES if feat in x]
            node_features = None

            if requested_node_features:
                node_features = []
                for i in range(batch_size):
                    features = []
                    for feat in requested_node_features:
                        v = x[feat][i].to(self.device, dtype=torch.float32)
                        if v.dim() == 1:
                            v = v.unsqueeze(-1)
                        features.append(v)

                    node_features.append(torch.cat(features, dim=-1))

            return self.core(x_vec, edges, node_count, node_features, edge_weights)

        if model_type in {"graph_neural_network", "graph_isomorphism_network"}:
            if "edges" not in x:
                raise KeyError(f"{model_type} requires 'edges' in input")
            if "node_count" not in x:
                raise KeyError(f"{model_type} requires 'node_count' in input")

            batch_size = x_vec.shape[0]
            node_count = x["node_count"].to(self.device)

            edges_list = []
            for i in range(batch_size):
                edges_mx2 = x["edges"][i].to(self.device)  # (M, 2)

                if "edge_weights" in x:
                    edge_weight = x["edge_weights"][i].to(self.device).unsqueeze(-1)
                    edges_mx2, edge_weight = _symmetrize_edges(edges_mx2, edge_weight)
                    edge_index = edges_mx2.t().contiguous()  # (2, 2M)
                    edges_list.append((edge_index, edge_weight))
                else:
                    edges_mx2, _ = _symmetrize_edges(edges_mx2)
                    edges_list.append(edges_mx2.t().contiguous())

            requested_node_features = [feat for feat in NODE_FEATURES if feat in x]
            node_features = None

            if requested_node_features:
                node_features = []
                for i in range(batch_size):
                    features = []
                    for feat in requested_node_features:
                        v = x[feat][i].to(self.device, dtype=torch.float32)
                        if v.dim() == 1:
                            v = v.unsqueeze(-1)
                        features.append(v)

                    node_features.append(torch.cat(features, dim=-1))

            if model_type == "graph_isomorphism_network":
                return self.core(x_vec, edges_list, node_count, node_features)

            requested_edge_features = [feat for feat in EDGE_FEATURES if feat in x]
            edge_features = None

            if requested_edge_features:
                edge_features = []
                for i in range(batch_size):
                    features = []
                    for feat in requested_edge_features:
                        v = x[feat][i].to(self.device, dtype=torch.float32)
                        if v.dim() == 1:
                            v = v.unsqueeze(-1)
                        features.append(v)

                    edge_features.append(torch.cat(features, dim=-1))

            return self.core(x_vec, edges_list, node_count, node_features, edge_features)

        if model_type == "gcn":
            if "edges" not in x:
                raise KeyError("gcn requires 'edges' in input")
            if "node_count" not in x:
                raise KeyError("gcn requires 'node_count' in input")

            node_count = x["node_count"].to(self.device)
            data_list = []

            for idx, ed in enumerate(x["edges"]):
                edges_mx2 = ed.to(self.device)  # (M, 2)
                num_nodes = int(node_count[idx])

                edge_weight = (
                    x["edge_weights"][idx].to(self.device).view(-1)
                    if "edge_weights" in x
                    else None
                )
                # Symmetrize edges 
                edges_mx2, edge_weight = _symmetrize_edges(edges_mx2, edge_weight)
                edge_index = edges_mx2.t().contiguous()  # (2, 2M)

                data_kwargs = dict(
                    x=torch.ones((num_nodes, 1), device=self.device),
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    features=x_vec[idx].unsqueeze(0),
                )

                if edge_weight is not None:
                        data_kwargs["edge_weight"] = edge_weight

                data_list.append(Data(**data_kwargs))

            return self.core(Batch.from_data_list(data_list).to(self.device))

        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: mlp, edge_transformer, diffusion_transformer, "
            f"graph_transformer, graph_neural_network, graph_isomorphism_network, gcn"
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.target_key not in y:
            raise KeyError(
                f"Missing target {self.target_key!r} in y keys={list(y.keys())}"
            )

        y_true_norm = y[self.target_key]
        if not torch.is_tensor(y_true_norm):
            y_true_norm = torch.tensor(y_true_norm)

        y_true_norm = y_true_norm.to(dtype=torch.float32, device=self.device)
        y_hat_norm = self.forward(x)
        loss = self.criterion(y_hat_norm, y_true_norm)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_hat_norm.shape[0],
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.core.parameters(), lr=float(self.hparams.lr))
        max_epochs = self.trainer.max_epochs
        warmup_epochs = 5

        # Cosine annealing scheduler with warmup
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linear warmup 
                return float(current_epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine annealing after warmup
                progress = float(current_epoch + 1 - warmup_epochs) / float(
                    max(1, max_epochs - warmup_epochs)
                )
                progress = min(progress, 1.0)

                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
