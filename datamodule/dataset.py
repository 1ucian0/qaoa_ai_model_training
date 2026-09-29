# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch to avoid segfault warning
# See: https://github.com/pytorch/pytorch/issues/78829
try:
    import juliacall
except ImportError:
    pass  # juliacall is optional

import json
import warnings
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset

from qaoa_training_pipeline.utils.graph_utils import load_graph
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator
from qaoa_training_pipeline.utils.graph_utils import solve_max_cut

from datamodule.utils import (
    angles_to_target,
    rescaling_factor,
    normalize_qaoa_params
)
from datamodule.features import (
    NODE_FEATURE_FUNCTIONS,
    EDGE_FEATURE_FUNCTIONS,
    GRAPH_FEATURE_FUNCTIONS,
)
from datamodule.meta_configs import (
    VALID_IN_FEATURES,
    VALID_OUT_FEATURES,
    SCALAR_FEATURES_IN,
    NODE_FEATURES,
    EDGE_FEATURES,
    GRAPH_FEATURES,
)

logger = logging.getLogger("[QAOA Dataset]")


class QAOADataset(Dataset):
    """
    Dataset for QAOA graph instances.
    """

    def __init__(
        self,
        summary_file: str | Path,
        instance_dir: str | Path,
        cut_metrics_dir: str | Path | None = None,
        instance_set: list[str] = ["random_regular"],
        in_features: list[str] | None = None,
        out_features: list[str] | None = None,
        target_p: int = 1,
        compute_cut_metrics: bool = False,
        num_timesteps: int = 0,
        use_edge_weights: bool = False,
    ) -> None:
        """
        Args:
        summary_file: Path to summary file.
        instance_dir: Path to instance directory.
        cut_metrics_dir: Path to cut metrics directory.
        instance_set: Instance directories to load from.
        in_features: List of input features to return in get_item.
        out_features: List of output features to return in get_item.
        target_p: Which QAOA depth p to load -> Filters instances based on that.
        compute_cut_metrics: Whether to compute max/min cut metrics.
        use_edge_weights: Whether to extract and include edge weights.
        """
        self.summary_file = Path(summary_file)
        self.instance_dir = Path(instance_dir)
        self.cut_metrics_dir = Path(cut_metrics_dir) if cut_metrics_dir else None
        self.instance_set = instance_set
        self.target_p = int(target_p)
        self.compute_cut_metrics = bool(compute_cut_metrics)
        self.norm_stats = None  # Variable to populate with normalization statistics
        self.num_timesteps = num_timesteps
        self.use_edge_weights = bool(use_edge_weights)

        if not (1 <= self.target_p <= 10):
            raise ValueError(f"target_p must be in [1,10], got {self.target_p}")

        requested_in = list(in_features) if in_features is not None else []
        requested_out = list(out_features) if out_features is not None else []
        invalid_in = [f for f in requested_in if f not in VALID_IN_FEATURES]
        invalid_out = [f for f in requested_out if f not in VALID_OUT_FEATURES]
        if invalid_in:
            warnings.warn(f"Ignoring invalid in_features={invalid_in}", stacklevel=2)
        if invalid_out:
            warnings.warn(f"Ignoring invalid out_features={invalid_out}", stacklevel=2)

        self.in_features = [f for f in requested_in if f in VALID_IN_FEATURES]
        self.out_features = [f for f in requested_out if f in VALID_OUT_FEATURES]

        if any(
            f in {"opt_max_cut", "opt_min_cut", "opt_approx_ratio"}
            for f in self.out_features
        ) and not (self.compute_cut_metrics or self.cut_metrics_dir):
            logger.warning(
                "Requested max-cut features but compute_cut_metrics=False; returning NaN for them."
            )

        self.samples = self._read_summary_for_p()

    def _read_summary_for_p(self) -> list[dict]:
        # Load summary file
        if not self.summary_file.exists():
            raise FileNotFoundError(self.summary_file)
        with self.summary_file.open("r") as fin:
            summary = json.load(fin)
        if not isinstance(summary, dict):
            raise ValueError("summary_file must contain a dict at top level")

        # Based on target_p and instance_set load instances for that QAOA depth
        p_key = str(self.target_p)
        out = []

        single_set = self.instance_set[0] if len(self.instance_set) == 1 else None

        for group_name, group_dict in summary.items():
            if not isinstance(group_dict, dict):
                continue

            if single_set is not None:
                instance_set = single_set
            elif group_name in self.instance_set:
                instance_set = group_name
            else:
                logger.debug(
                    "Skipping summary group %r: not in instance_set=%r. When a "
                    "summary covers several instance sets, name each top-level "
                    "group after its set.",
                    group_name,
                    self.instance_set,
                )
                continue

            for instance_key, p_dict in group_dict.items():
                if not isinstance(p_dict, dict) or p_key not in p_dict:
                    continue

                entry = p_dict[p_key]
                if not isinstance(entry, dict):
                    continue

                instance_id = Path(instance_key).stem

                angles = entry.get("qaoa_angles")
                if not isinstance(angles, list):
                    raise ValueError(
                        f"qaoa_angles must be a list for {instance_key} p={p_key}"
                    )
                if len(angles) != 2 * self.target_p:
                    raise ValueError(
                        f"{instance_id}: qaoa_angles len {len(angles)} != 2*p ({2 * self.target_p})"
                    )

                result_file = entry.get("result_file_name")
                if not isinstance(result_file, str):
                    raise ValueError(
                        f"result_file_name must be str for {instance_key} p={p_key}"
                    )

                record = {
                    "instance_set": str(instance_set),
                    "instance_id": str(instance_id),
                    "depth_p": int(self.target_p),
                    "opt_energy": float(entry["energy"]),
                    "qaoa_angles": [float(a) for a in angles],
                    "result_file_name": str(result_file),
                }

                out.append(record)

        out.sort(key=lambda r: r["instance_id"])

        logger.info(f"Loaded {len(out)} samples for p={self.target_p}, instance_sets={self.instance_set}")
        if not out:
            raise ValueError(f"No samples found for target_p={self.target_p}")
        return out

    @lru_cache(
        maxsize=256
    ) 
    def _load_graph(
        self, instance_id: str, instance_set: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load graph from instance_id (e.g. from QAOA Parameter Setting Repo)
        """
        graph_path = self.instance_dir / instance_set / f"{instance_id}.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph instance path not found: {graph_path}")
        return load_graph(graph_path)

    def _load_cut_metrics(
        self, instance_id: str, instance_set: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load cut metrics from instance_id.
        """
        cut_metrics_path = (
            self.cut_metrics_dir / instance_set / f"{instance_id}.json"
        )
        if not cut_metrics_path.exists():
            raise FileNotFoundError(f"Cut metrics path not found: {cut_metrics_path}")

        with open(cut_metrics_path, "r") as cut_metrics:
            cut_metrics_dict = json.load(cut_metrics)

        return cut_metrics_dict["max_cut"], cut_metrics_dict["min_cut"]

    def __len__(self) -> int:
        return len(self.samples)

    def compute_normalizer_stats(self, indices: list[int]) -> dict:
        """
        Compute normalization statistics (mean, std) on a subset of the dataset.

        - Uses raw (unnormalized) values from samples
        - Only considers scalar features listed in SCALAR_FEATURES_IN
        - Optimized to avoid loading graphs when only basic scalar features are needed
        - Returns stats in the form:
            {
              "x": {feature: (mean, std)},
              "y": {feature: (mean, std)},
            }
        """
        feats_in = [f for f in self.in_features if f in SCALAR_FEATURES_IN]

        x_vals = {f: [] for f in feats_in}

        # Log what features we're computing stats for
        if feats_in:
            logger.info(
                f"Computing normalization stats for {len(feats_in)} input features: {feats_in}"
            )

        # Check if we need to load graphs for any requested features
        graph_derived_features = {
            "num_nodes",
            "num_edges",
            "edges_per_node",
            "mean_degree",
            "std_degree",
        }
        needs_graph_for_stats = any(f in graph_derived_features for f in feats_in)
        needs_graph_features = any(f in GRAPH_FEATURES for f in feats_in)

        # If we need graph-level features (like graph_diameter), we must load the full graph
        if needs_graph_features:
            # Fall back to loading full items for complex graph features
            expensive_features = [f for f in feats_in if f in GRAPH_FEATURES]
            logger.warning(
                f"Computing normalization stats with {len(expensive_features)} expensive graph-level features: {expensive_features}. "
                f"This will be SLOW (~{len(indices)} graphs × expensive computations). "
                f"Consider: (1) Pre-computing and saving norm_stats, or (2) Removing these features from in_features if not needed."
            )
            for i in indices:
                x, y = self._raw_item(i)

                for f in feats_in:
                    if f in x:
                        x_vals[f].append(float(x[f]))
        else:
            # Optimized path for basic features
            logger.info("Computing normalization stats (optimized path)...")
            for i in indices:
                s = self.samples[i]

                # Collect scalar input features
                x_scalars = {}

                # Only load graph if needed for graph-derived scalar features
                if needs_graph_for_stats:
                    g = self._load_graph(s["instance_id"], s["instance_set"])
                    if "num_nodes" in feats_in:
                        x_scalars["num_nodes"] = int(g.number_of_nodes())
                    if "num_edges" in feats_in:
                        x_scalars["num_edges"] = int(g.number_of_edges())
                    if "edges_per_node" in feats_in:
                        num_nodes = x_scalars.get("num_nodes", int(g.number_of_nodes()))
                        num_edges = x_scalars.get("num_edges", int(g.number_of_edges()))
                        x_scalars["edges_per_node"] = num_edges / num_nodes
                    if "mean_degree" in feats_in or "std_degree" in feats_in:
                        degrees = np.asarray([d for _n, d in g.degree()])
                        if "mean_degree" in feats_in:
                            x_scalars["mean_degree"] = float(np.mean(degrees))
                        if "std_degree" in feats_in:
                            x_scalars["std_degree"] = float(np.std(degrees))

                # Collect input features
                for f in feats_in:
                    if f in x_scalars:
                        x_vals[f].append(float(x_scalars[f]))

        def mean_std(vals):
            arr = np.asarray(vals, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std())
            return mean, (std if std > 1e-12 else 1.0)

        stats = {
            "x": {f: mean_std(v) for f, v in x_vals.items() if v},
            "y": {},
        }

        return stats

    def validate_normalizer_stats(self, stats: dict) -> None:
        """
        Validate normalization statistics.

        Ensures that:
          - all scalar input features have stats in stats["x"]
        """
        if not isinstance(stats, dict):
            raise ValueError("norm_stats must be a dict")

        expected_x = {f for f in self.in_features if f in SCALAR_FEATURES_IN}
        provided_x = set(stats.get("x", {}).keys())
        missing_x = expected_x - provided_x

        if missing_x:
            raise ValueError(
                f"Normalization stats missing X features: {sorted(missing_x)}"
            )

    def set_normalizer_stats(self, stats: dict | None) -> None:
        self.validate_normalizer_stats(stats)
        self.norm_stats = stats

    def _raw_item(self, idx: int) -> tuple[dict, torch.Tensor]:

        s = self.samples[idx]

        g = self._load_graph(s["instance_id"], s["instance_set"])

        # node_count comes from the graph itself rather than from the filename, so
        # instance names carry no structural meaning and can follow any convention.
        x = {"node_count": g.number_of_nodes(), "depth_p": s["depth_p"]}

        # Capture original sum of weights before rescaling (for cut metrics calculation)
        original_sum_weights = sum(
            data.get("weight", 1.0) for _, _, data in g.edges(data=True)
        )

        cost_op = graph_to_operator(g, pre_factor=-0.5)
        rescale_a = rescaling_factor(cost_op)
        x["rescale_a"] = rescale_a
        qaoa_params = angles_to_target(
            s["qaoa_angles"], s["depth_p"], rescale_a
        )

        y = {
            "qaoa_params": qaoa_params,
            "opt_energy": s["opt_energy"],
        }

        x["nodes"] = list(g.nodes())

        # Extract edges as a (num_edges, 2) tensor (variable length per graph)
        edge_list = list(g.edges())
        x["edges"] = torch.as_tensor(edge_list, dtype=torch.long)

        # Extract edge weights only if requested
        if self.use_edge_weights:
            # Extract edge weights with rescaling (variable length per graph)
            edge_weights = [
                g.edges[u, v].get("weight", 1.0) / rescale_a for u, v in edge_list
            ]
            x["edge_weights"] = torch.as_tensor(
                edge_weights, dtype=torch.float32
            )

        x["num_nodes"] = int(g.number_of_nodes())
        x["num_edges"] = int(g.number_of_edges())
        x["edges_per_node"] = x["num_edges"] / x["num_nodes"]
        degrees = np.asarray([d for _n, d in g.degree()])
        x["mean_degree"] = np.mean(degrees)
        x["std_degree"] = np.std(degrees)
        x["t"] = (
            torch.randint(0, self.num_timesteps, size=())
            if self.num_timesteps
            else 0
        )

        # Store metadata for graph reconstruction with weights
        x["instance_id"] = s["instance_id"]
        x["instance_set"] = s["instance_set"]

        cut_metrics_path = (
            self.cut_metrics_dir / s["instance_set"] / f"{s['instance_id']}.json"
            if self.cut_metrics_dir
            else None
        )
        if cut_metrics_path is not None and cut_metrics_path.exists():
            max_cut, min_cut = self._load_cut_metrics(
                s["instance_id"], s["instance_set"]
            )

            # opt_energy is for the ORIGINAL graph, max_cut/min_cut are for ORIGINAL graph
            # But g has been rescaled, so we use the original_sum_weights captured before rescaling
            cut_val = s["opt_energy"] + 0.5 * original_sum_weights
            y["opt_max_cut"] = max_cut
            y["opt_min_cut"] = min_cut
            y["opt_approx_ratio"] = float((cut_val - min_cut) / (max_cut - min_cut))

        elif self.compute_cut_metrics:
            # Reuse the cost_op already built above (same pre_factor=-0.5).
            max_cut, min_cut, approx_ratio = solve_max_cut(cost_op, s["opt_energy"])
            y["opt_max_cut"] = float(max_cut)
            y["opt_min_cut"] = float(min_cut)
            y["opt_approx_ratio"] = float(approx_ratio)
        else:
            y["opt_max_cut"] = float("nan")
            y["opt_min_cut"] = float("nan")
            y["opt_approx_ratio"] = float("nan")

        # === Compute requested node features ===
        requested_node_features = [
            f for f in self.in_features if f in NODE_FEATURES
        ]

        if requested_node_features:
            for feat_name in requested_node_features:
                func = NODE_FEATURE_FUNCTIONS[feat_name]

                # Handle special parameter requirements for alternative features
                if feat_name == "node_degree_zscore":
                    # Requires mean_deg and std_deg (already computed above)
                    feat_values = func(
                        g, x["num_nodes"], x["mean_degree"], x["std_degree"]
                    )
                else:
                    # Standard node features with (graph, num_nodes) signature
                    feat_values = func(g, x["num_nodes"])

                # Check for NaN/Inf and replace with 0.0
                if np.isnan(feat_values).any() or np.isinf(feat_values).any():
                    logger.warning(
                        f"NaN/Inf in {feat_name} for {s['instance_id']}, replacing with 0.0"
                    )
                    feat_values = np.nan_to_num(
                        feat_values, nan=0.0, posinf=0.0, neginf=0.0
                    )
                x[feat_name] = torch.from_numpy(feat_values)

        # === Compute requested edge features ===
        requested_edge_features = [
            f for f in self.in_features if f in EDGE_FEATURES
        ]

        if requested_edge_features:
            for feat_name in requested_edge_features:
                func = EDGE_FEATURE_FUNCTIONS[feat_name]
                feat_values = func(g, edge_list)
                x[feat_name] = torch.from_numpy(feat_values)  # (num_edges,)

        # === Compute requested graph-level features ===
        requested_graph_features = [
            f for f in self.in_features if f in GRAPH_FEATURES
        ]

        if requested_graph_features:
            for feat_name in requested_graph_features:
                func = GRAPH_FEATURE_FUNCTIONS[feat_name]
                feat_value = func(g)
                # Check for NaN/Inf and replace with 0.0
                if np.isnan(feat_value) or np.isinf(feat_value):
                    logger.warning(
                        f"NaN/Inf in {feat_name} for {s['instance_id']}, replacing with 0.0"
                    )
                    feat_value = 0.0
                x[feat_name] = float(feat_value)  # scalar

        return x, y

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        x_all, y_all = self._raw_item(idx)

        x, y = {}, {}
        nx = (self.norm_stats or {}).get("x", {})

        def to_tensor(v):
            return (
                torch.tensor(float(v), dtype=torch.float32)
                if isinstance(v, (int, float, np.integer, np.floating))
                else v
            )

        for k, v in x_all.items():
            if k in self.in_features:
                # Keep string fields as-is (instance_id, instance_set)
                if isinstance(v, str):
                    x[k] = v
                else:
                    if k in nx:
                        mean, std = nx[k]
                        std_safe = max(std, 1e-6)
                        v = (float(v) - mean) / std_safe
                    x[k] = to_tensor(v)

        for k, v in y_all.items():
            if k in self.out_features:
                y[k] = to_tensor(v)

        if self.norm_stats:
            y["qaoa_params"] = normalize_qaoa_params(y["qaoa_params"])

        return x, y
