# Copyright IBM Corp. 2026
# License: Apache-2.0

from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import torch
import json
import warnings

from qaoa_training_pipeline.evaluation.efficient_depth_one import (
    EfficientDepthOneEvaluator,
)
from qaoa_training_pipeline.evaluation import PPEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator, load_graph

from datamodule.utils import undo_gamma_rescale, denormalize_qaoa_params

# Per-worker evaluators, populated by _pool_init in each spawned process.
_WORKER_EVAL_DEPTH1 = None
_WORKER_EVAL_PP = None


def _pool_init() -> None:
    """Initializer run once in each pool worker process.

    Builds the depth-1 and p>=2 evaluators a single time per worker so the
    Julia warmup for PPEvaluator is amortised across every instance that
    worker handles.
    """
    global _WORKER_EVAL_DEPTH1, _WORKER_EVAL_PP
    _WORKER_EVAL_DEPTH1 = EfficientDepthOneEvaluator()
    _WORKER_EVAL_PP = PPEvaluator({"max_weight": 4, "min_abs_coeff": 1e-5})


def _eval_one_instance(
    instance_id: str,
    instance_set: str,
    instance_dir: str,
    betas: list[float],
    gammas: list[float],
    max_cut: float,
    min_cut: float,
    depth: int,
    evaluator_depth1=None,
    evaluator_pp=None,
) -> tuple[float, float, float, float]:
    """Compute QAOA energy + approximation ratio for one instance.

    Pure function so it can run in a pool worker. Loads the graph
    from disk, converts to a cost operator, evaluates the energy with the
    depth-appropriate evaluator, and returns ``(energy, approx_ratio, max_cut,
    min_cut)`` matching ``QAOAEvalMixin.evaluate_qaoa``.
    """
    if evaluator_depth1 is None:
        evaluator_depth1 = _WORKER_EVAL_DEPTH1
    if evaluator_pp is None:
        evaluator_pp = _WORKER_EVAL_PP

    file_instance_id = str(instance_id)
    graph_path = Path(instance_dir) / instance_set / f"{file_instance_id}.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph instance path not found: {graph_path}")
    graph = load_graph(graph_path)
    cost_op = graph_to_operator(graph, pre_factor=-0.5)

    if int(depth) == 1:
        beta = float(np.asarray(betas).reshape(-1)[0])
        gamma = float(np.asarray(gammas).reshape(-1)[0])
        energy = evaluator_depth1.evaluate(cost_op=cost_op, params=[beta, gamma])
    else:
        betas_l = np.asarray(betas).reshape(-1).astype(float).tolist()
        gammas_l = np.asarray(gammas).reshape(-1).astype(float).tolist()
        energy = evaluator_pp.evaluate(cost_op=cost_op, params=[*betas_l, *gammas_l])

    weights = [val[2].get("weight", 1.0) for val in graph.edges(data=True)]
    sum_weights = sum(weights)
    cut_val = float(energy) + 0.5 * float(sum_weights)

    denom = float(max_cut) - float(min_cut)
    if abs(denom) < 1e-12:
        approximation_ratio = 0.0
    else:
        approximation_ratio = (cut_val - float(min_cut)) / denom

    return float(energy), float(approximation_ratio), float(max_cut), float(min_cut)


class QAOAEvalMixin:
    """
    Mixin for the validation and test step and logging.
    """

    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self._eval_depth1 = EfficientDepthOneEvaluator()
        self._eval_pp = PPEvaluator({"max_weight": 4, "min_abs_coeff": 1e-5})

        # Cache for loaded graphs and cost operators, keyed by (instance_set, instance_id).
        self._graph_cache = {}
        self._cost_op_cache = {}

    def load_graph_with_weights(
        self, instance_id: str, instance_set: str, instance_dir: Path
    ) -> nx.Graph:
        """Load graph from file to get edge weights (needed for heavy_hex and other weighted graphs)"""
        cache_key = (str(instance_set), str(instance_id))
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        file_instance_id = str(instance_id)
        graph_path = instance_dir / instance_set / f"{file_instance_id}.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph instance path not found: {graph_path}")
        graph = load_graph(graph_path)
        self._graph_cache[cache_key] = graph
        return graph

    def _graph_to_cost_op(self, graph: nx.Graph, key: tuple | None = None):
        """Convert a graph to a cost operator, caching per (instance_set, instance_id).

        Instances are fixed across epochs, so each conversion happens once.
        """
        if key is not None and key in self._cost_op_cache:
            return self._cost_op_cache[key]
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
        if key is not None:
            self._cost_op_cache[key] = cost_op
        return cost_op

    def evaluate_qaoa(
        self,
        betas_i: np.ndarray,
        gammas_i: np.ndarray,
        max_cut: float | torch.Tensor,
        min_cut: float | torch.Tensor,
        instance_id: str | None = None,
        instance_set: str | None = None,
        instance_dir: Path | None = None,
    ) -> tuple[float, float, float, float]:

        graph = self.load_graph_with_weights(instance_id, instance_set, instance_dir)
        cost_op_key = (str(instance_set), str(instance_id))
        cost_op = self._graph_to_cost_op(graph, cost_op_key)

        max_cut = (
            float(max_cut.detach().cpu().item())
            if torch.is_tensor(max_cut)
            else float(max_cut)
        )
        min_cut = (
            float(min_cut.detach().cpu().item())
            if torch.is_tensor(min_cut)
            else float(min_cut)
        )

        if int(self.depth) == 1:
            evaluator = self._eval_depth1
            beta = float(np.asarray(betas_i).reshape(-1)[0])
            gamma = float(np.asarray(gammas_i).reshape(-1)[0])
            energy = evaluator.evaluate(cost_op=cost_op, params=[beta, gamma])
        else:
            evaluator = self._eval_pp
            betas = np.asarray(betas_i).reshape(-1).astype(float).tolist()
            gammas = np.asarray(gammas_i).reshape(-1).astype(float).tolist()
            params = [*betas, *gammas]
            energy = evaluator.evaluate(cost_op=cost_op, params=params)

        weights = [val[2].get("weight", 1.0) for val in graph.edges(data=True)]
        sum_weights = sum(weights)
        cut_val = float(energy) + 0.5 * float(sum_weights)

        denom = max_cut - min_cut
        if abs(denom) < 1e-12:
            approximation_ratio = 0.0
        else:
            approximation_ratio = (cut_val - min_cut) / denom

        return float(energy), float(approximation_ratio), float(max_cut), float(min_cut)

    def compute_mean_std(self, values: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        if not values:
            z = torch.tensor(0.0, device=self.device)
            return z, z

        t = torch.tensor(values, dtype=torch.float32, device=self.device)
        return t.mean(), t.std(unbiased=False)

    def compute_tail_stats(
        self, values: list[float], near_threshold: float = 0.999
    ) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "p5": 0.0, "p10": 0.0, "frac_near": 0.0}

        arr = np.asarray(values, dtype=np.float64)
        return {
            "min": float(arr.min()),
            "p5": float(np.percentile(arr, 5)),
            "p10": float(np.percentile(arr, 10)),
            "frac_near": float((arr >= near_threshold).mean()),
        }

    def _load_fixed_angle_results(
        self, x: dict[str, any], y: dict[str, any], i: int
    ) -> tuple[float | None, list, list]:
        """
        Load fixed angle QAOA results from JSON file.

        Args:
            x: Input batch dictionary containing instance_set and instance_id
            y: Target batch dictionary containing opt_max_cut and opt_min_cut
            i: Sample index within batch
        """
        fa_instance_id = str(x["instance_id"][i])
        file_path = (
            self.fixed_angle_dir
            / f"p_{self.depth}"
            / x["instance_set"][i]
            / f"{fa_instance_id}.json"
        )

        if file_path.exists():
            # Load pre-computed fixed angle results from JSON
            with open(file_path) as fp:
                data = json.load(fp)

            cut_value = data.get("fixed_angle_cut")
            fa_betas, fa_gammas = data.get("betas"), data.get("gammas")

            # Compute approximation ratio from cut value
            denom = (
                y["opt_max_cut"][i].detach().cpu() - y["opt_min_cut"][i].detach().cpu()
            )
            if abs(denom) < 1e-12:
                fa_ratio = 0.0
            else:
                fa_ratio = (cut_value - y["opt_min_cut"][i].detach().cpu()) / denom

            fixed_angle_ar = float(fa_ratio)
        else:
            # return None to indicate missing data
            warnings.warn(
                "Fixed Angle json not found. Fixed angle AR and parameters will not be available for comparison."
            )
            fixed_angle_ar = None
            fa_betas, fa_gammas = None, None

        return fixed_angle_ar, fa_betas, fa_gammas

    def _evaluate_and_log_sample(
        self,
        x: dict[str, any],
        y: dict[str, any],
        y_hat: torch.Tensor,
        y_true_cpu: torch.Tensor,
        i: int,
        metrics_dict: dict[str, list],
        params_dict: dict[str, list],
        precomputed: tuple[float, float] | None = None,
        per_instance: list[dict] | None = None,
    ) -> None:
        """
        Shared logic for evaluating and logging a sample during validation or test.

        Args:
            x: Input batch dictionary containing edges, node_count, instance_id, instance_set
            y: Target batch dictionary containing opt_max_cut, opt_min_cut, opt_approx_ratio
            y_hat: Predicted QAOA parameters (denormalized, on CPU)
            y_true_cpu: True QAOA parameters (denormalized, on CPU)
            i: Sample index within batch
            metrics_dict: Dictionary to store global metrics (energy, approx_ratio, etc.)
            params_dict: Dictionary to store global parameters (beta, gamma, etc.)
            precomputed: Optional ``(energy, approx_ratio)`` already computed for
                this sample (e.g. by the test-time process pool). When provided,
                the in-line classical evaluation is skipped.
        """
        edges_i = x["edges"][i]
        betas_i = y_hat[i, : self.depth]
        gammas_i = y_hat[i, self.depth : 2 * self.depth]

        opt_betas_i = y_true_cpu[i, : self.depth]
        opt_gammas_i = y_true_cpu[i, self.depth : 2 * self.depth]

        # Evaluate QAOA with predicted parameters (or use the pooled result)
        if precomputed is not None:
            energy, approx_ratio = precomputed
        else:
            energy, approx_ratio, _, _ = self.evaluate_qaoa(
                betas_i=betas_i.numpy(),
                gammas_i=gammas_i.numpy(),
                max_cut=y["opt_max_cut"][i].detach().cpu(),
                min_cut=y["opt_min_cut"][i].detach().cpu(),
                instance_id=x["instance_id"][i] if "instance_id" in x else None,
                instance_set=x["instance_set"][i] if "instance_set" in x else None,
                instance_dir=self.instance_dir,
            )

        opt_approx_ratio = y["opt_approx_ratio"][i].detach().cpu()

        # Load fixed angle results
        fixed_angle_ar, fa_betas, fa_gammas = self._load_fixed_angle_results(
            x, y, i
        )

        # Log fixed angle parameters
        if fa_betas is not None and fa_gammas is not None:
            for j in range(self.depth):
                params_dict[f"fixed_angle_beta_{j}"].append(float(fa_betas[j]))
                params_dict[f"fixed_angle_gamma_{j}"].append(float(fa_gammas[j]))

        # Reach: model AR as a fraction of the optimal-angle AR for this
        # instance. reach=1.0 means the prediction matched the optimal angles on this instance.
        opt_ar = float(opt_approx_ratio)
        reach = float(approx_ratio) / opt_ar if abs(opt_ar) > 1e-12 else 0.0

        # Per-instance record: one row per test instance, written to test_per_instance.csv.
        if per_instance is not None:
            instance_set_value = (
                str(x["instance_set"][i]) if "instance_set" in x else "unknown"
            )
            row = {
                "instance_id": str(x["instance_id"][i])
                if "instance_id" in x
                else "unknown",
                "instance_set": instance_set_value,
                "graph_type": instance_set_value,
                "node_count": int(x["node_count"][i])
                if "node_count" in x
                else -1,
                "num_edges": int(edges_i.shape[0]),
                "depth_p": int(self.depth),
                "reach": reach,
                "approx_ratio": float(approx_ratio),
                "opt_approx_ratio": opt_ar,
                "energy": float(energy),
                "fixed_angle_approx_ratio": float(fixed_angle_ar)
                if fixed_angle_ar is not None
                else None,
            }
            for j in range(self.depth):
                row[f"beta_{j}"] = float(betas_i[j])
                row[f"gamma_{j}"] = float(gammas_i[j])
            per_instance.append(row)

        # Log global metrics
        metrics_dict["energy"].append(float(energy))
        metrics_dict["approx_ratio"].append(float(approx_ratio))
        metrics_dict["opt_approx_ratio"].append(opt_approx_ratio)
        metrics_dict["reach"].append(reach)
        if fixed_angle_ar is not None:
            metrics_dict["fixed_angle_approx_ratio"].append(float(fixed_angle_ar))

        # Log predicted and optimal parameters
        for j in range(self.depth):
            params_dict[f"beta_{j}"].append(betas_i[j])
            params_dict[f"gamma_{j}"].append(gammas_i[j])
            params_dict[f"opt_beta_{j}"].append(opt_betas_i[j])
            params_dict[f"opt_gamma_{j}"].append(opt_gammas_i[j])

    def _reset_validation_accumulators(self) -> None:
        self._val_metrics = {
            "energy": [],
            "approx_ratio": [],
            "opt_approx_ratio": [],
            "reach": [],
            "fixed_angle_approx_ratio": [],
        }

        self._val_params = {}
        self._val_params.update({f"beta_{i}": [] for i in range(self.depth)})
        self._val_params.update({f"gamma_{i}": [] for i in range(self.depth)})
        self._val_params.update({f"opt_beta_{i}": [] for i in range(self.depth)})
        self._val_params.update({f"opt_gamma_{i}": [] for i in range(self.depth)})
        self._val_params.update(
            {f"fixed_angle_beta_{i}": [] for i in range(self.depth)}
        )
        self._val_params.update(
            {f"fixed_angle_gamma_{i}": [] for i in range(self.depth)}
        )

    def on_validation_epoch_start(self) -> None:
        self._reset_validation_accumulators()

    def validation_step(
        self, batch: tuple[dict[str, any], dict[str, any]], batch_idx: int
    ) -> None:
        x, y = batch

        if self.target_key not in y:
            raise KeyError(
                f"Missing target {self.target_key!r} in y keys={list(y.keys())}"
            )

        if "edges" not in x:
            raise KeyError("Validation requires 'edges' in x for QAOA evaluation.")
        if "node_count" not in x:
            raise KeyError("Validation requires 'node_count' in x for grouping.")

        y_true_norm = y[self.target_key]
        y_true = denormalize_qaoa_params(y_true_norm)
        y_true = undo_gamma_rescale(y_true, p=self.depth, rescale_a=x["rescale_a"])
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(y_true)
        y_true = y_true.to(dtype=torch.float32, device=self.device)

        y_hat_norm = self.forward(x)
        y_hat = denormalize_qaoa_params(y_hat_norm)
        y_hat = undo_gamma_rescale(y_hat, p=self.depth, rescale_a=x["rescale_a"])

        y_true_norm = y_true_norm.to(dtype=torch.float32, device=self.device)
        loss = self.criterion(y_hat_norm, y_true_norm)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_hat.shape[0],
        )

        # Validation approximation-ratio / energy evaluation is computed only for
        # p==1 (cheap EfficientDepthOneEvaluator).  TODO maybe change back
        if int(self.depth) != 1:
            return

        y_hat = y_hat.detach().cpu()
        y_true_cpu = y_true.detach().cpu()
        batch_size = y_hat.shape[0]

        for i in range(batch_size):
            # Use shared evaluation and logging method
            self._evaluate_and_log_sample(
                x=x,
                y=y,
                y_hat=y_hat,
                y_true_cpu=y_true_cpu,
                i=i,
                metrics_dict=self._val_metrics,
                params_dict=self._val_params,
            )

    def on_validation_epoch_end(self) -> None:
        # For p>=2 validation logs only val/loss (no per-sample QAOA metrics) TODO maybe change back
        if int(self.depth) != 1:
            return

        count = len(self._val_metrics["energy"])

        count_t = torch.tensor(float(count), device=self.device)
        assert count_t.item() > 0, "No validation samples processed"
        self.log("val/count", count_t, prog_bar=False,)
        self.log(
            "val/fixed_angle_count",
            float(len(self._val_metrics["fixed_angle_approx_ratio"])),
            prog_bar=False,
        )

        energy_mean, energy_std = self.compute_mean_std(self._val_metrics["energy"])
        ratio_mean, ratio_std = self.compute_mean_std(self._val_metrics["approx_ratio"])
        opt_ratio_mean, opt_ratio_std = self.compute_mean_std(
            self._val_metrics["opt_approx_ratio"]
        )
        fixed_angle_ratio_mean, fixed_angle_ratio_std = self.compute_mean_std(
            self._val_metrics["fixed_angle_approx_ratio"]
        )
        reach_mean, reach_std = self.compute_mean_std(self._val_metrics["reach"])
        reach_tail = self.compute_tail_stats(self._val_metrics["reach"])

        self.log("val/approx_ratio_mean", ratio_mean, prog_bar=True,)
        self.log("val/approx_ratio_std", ratio_std, prog_bar=False,)
        self.log(
            "val/opt_approx_ratio_mean", opt_ratio_mean, prog_bar=True,
        )
        self.log(
            "val/opt_approx_ratio_std", opt_ratio_std, prog_bar=False,
        )
        self.log("val/reach_mean", reach_mean, prog_bar=True,)
        self.log("val/reach_std", reach_std, prog_bar=False,)
        self.log("val/reach_min", reach_tail["min"], prog_bar=False, sync_dist=False)
        self.log("val/reach_p5", reach_tail["p5"], prog_bar=False, sync_dist=False)
        self.log("val/reach_p10", reach_tail["p10"], prog_bar=False, sync_dist=False)
        self.log(
            "val/reach_frac_ge_999",
            reach_tail["frac_near"],
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "val/fixed_angle_approx_ratio_mean",
            fixed_angle_ratio_mean,
            prog_bar=True,
        )
        self.log(
            "val/fixed_angle_approx_ratio_std",
            fixed_angle_ratio_std,
            prog_bar=False,
        )

        for i in range(self.depth):
            beta_mean, beta_std = self.compute_mean_std(self._val_params[f"beta_{i}"])
            gamma_mean, gamma_std = self.compute_mean_std(
                self._val_params[f"gamma_{i}"]
            )
            opt_beta_mean, opt_beta_std = self.compute_mean_std(
                self._val_params[f"opt_beta_{i}"]
            )
            opt_gamma_mean, opt_gamma_std = self.compute_mean_std(
                self._val_params[f"opt_gamma_{i}"]
            )

            self.log(
                f"val/beta_{i + 1}_mean", beta_mean, prog_bar=False,
            )
            self.log(f"val/beta_{i + 1}_std", beta_std, prog_bar=False,)
            self.log(
                f"val/gamma_{i + 1}_mean", gamma_mean, prog_bar=False,
            )
            self.log(
                f"val/gamma_{i + 1}_std", gamma_std, prog_bar=False,
            )
            self.log(
                f"val/opt_beta_{i + 1}_mean",
                opt_beta_mean,
                prog_bar=False,
            )
            self.log(
                f"val/opt_beta_{i + 1}_std",
                opt_beta_std,
                prog_bar=False,
            )
            self.log(
                f"val/opt_gamma_{i + 1}_mean",
                opt_gamma_mean,
                prog_bar=False,
            )
            self.log(
                f"val/opt_gamma_{i + 1}_std",
                opt_gamma_std,
                prog_bar=False,
                
            )

    def _maybe_start_eval_pool(self) -> None:
        """Create a persistent 'spawn' process pool for classical energy eval.

        No-op unless ``self.test_eval_workers > 1``. Uses the "spawn" start
        method (juliacall is unsafe to fork) and initialises each worker's
        evaluators once via ``_pool_init``.
        """
        self._eval_pool = None
        workers = int(getattr(self, "test_eval_workers", 1) or 1)
        if workers <= 1:
            return
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        self._eval_pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_pool_init
        )

    def _shutdown_eval_pool(self) -> None:
        pool = getattr(self, "_eval_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._eval_pool = None

    def _compute_batch_energies(
        self, x: dict[str, any], y: dict[str, any], y_hat: torch.Tensor
    ) -> list[tuple[float, float]]:
        """Compute ``(energy, approx_ratio)`` for every sample in the batch.

        When a process pool is active the per-instance classical evaluations are
        dispatched concurrently.
        """
        batch_size = y_hat.shape[0]
        instance_dir = str(self.instance_dir)
        args = []
        for i in range(batch_size):
            betas_i = y_hat[i, : self.depth].numpy().reshape(-1).tolist()
            gammas_i = y_hat[i, self.depth : 2 * self.depth].numpy().reshape(-1).tolist()
            args.append(
                (
                    str(x["instance_id"][i]),
                    str(x["instance_set"][i]),
                    instance_dir,
                    betas_i,
                    gammas_i,
                    float(y["opt_max_cut"][i].detach().cpu()),
                    float(y["opt_min_cut"][i].detach().cpu()),
                    int(self.depth),
                )
            )

        pool = getattr(self, "_eval_pool", None)
        if pool is not None:
            results = list(pool.map(_eval_one_instance, *zip(*args)))
        else:
            results = [
                _eval_one_instance(
                    *a,
                    evaluator_depth1=self._eval_depth1,
                    evaluator_pp=self._eval_pp,
                )
                for a in args
            ]
        return [(energy, ar) for (energy, ar, _, _) in results]

    def on_test_epoch_start(self) -> None:
        self._maybe_start_eval_pool()
        self._test_per_instance = []

        self._test_metrics = {
            "energy": [],
            "approx_ratio": [],
            "opt_approx_ratio": [],
            "reach": [],
            "fixed_angle_approx_ratio": [],
        }
        self._test_params = {}
        self._test_params.update({f"beta_{i}": [] for i in range(self.depth)})
        self._test_params.update({f"gamma_{i}": [] for i in range(self.depth)})
        self._test_params.update({f"opt_beta_{i}": [] for i in range(self.depth)})
        self._test_params.update({f"opt_gamma_{i}": [] for i in range(self.depth)})
        self._test_params.update(
            {f"fixed_angle_beta_{i}": [] for i in range(self.depth)}
        )
        self._test_params.update(
            {f"fixed_angle_gamma_{i}": [] for i in range(self.depth)}
        )

    def test_step(
        self, batch: tuple[dict[str, any], dict[str, any]], batch_idx: int
    ) -> None:
        
        x, y = batch
        if self.target_key not in y:
            raise KeyError(
                f"Missing target {self.target_key!r} in y keys={list(y.keys())}"
            )

        if "edges" not in x:
            raise KeyError("Testing requires 'edges' in x for QAOA evaluation.")
        if "node_count" not in x:
            raise KeyError("Testing requires 'node_count' in x for grouping.")

        y_true_norm = y[self.target_key]
        y_true = denormalize_qaoa_params(y_true_norm)
        y_true = undo_gamma_rescale(y_true, p=self.depth, rescale_a=x["rescale_a"])
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(y_true)
        y_true = y_true.to(dtype=torch.float32, device=self.device)

        y_hat_norm = self.forward(x)
        y_hat = denormalize_qaoa_params(y_hat_norm)
        y_hat = undo_gamma_rescale(y_hat, p=self.depth, rescale_a=x["rescale_a"])
        y_true_norm = y_true_norm.to(dtype=torch.float32, device=self.device)
        loss = self.criterion(y_hat_norm, y_true_norm)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_hat.shape[0],
        )

        y_hat = y_hat.detach().cpu()
        y_true_cpu = y_true.detach().cpu()
        batch_size = y_hat.shape[0]

        precomputed = None
        pool_active = getattr(self, "_eval_pool", None) is not None
        has_instance_meta = (
            "instance_id" in x
            and "instance_set" in x
            and self.instance_dir is not None
            and x["instance_id"][0] != "unknown"
        )
        if pool_active and has_instance_meta:
            precomputed = self._compute_batch_energies(x, y, y_hat)

        for i in range(batch_size):
            self._evaluate_and_log_sample(
                x=x,
                y=y,
                y_hat=y_hat,
                y_true_cpu=y_true_cpu,
                i=i,
                metrics_dict=self._test_metrics,
                params_dict=self._test_params,
                per_instance=self._test_per_instance,
                precomputed=precomputed[i] if precomputed is not None else None,
            )

    def on_test_epoch_end(self) -> None:

        count = len(self._test_metrics["energy"])

        count_t = torch.tensor(float(count), device=self.device)
        assert count_t.item() > 0, "No test samples processed"
        self.log("test/count", count_t, prog_bar=False,)
        self.log(
            "test/fixed_angle_count",
            float(len(self._test_metrics["fixed_angle_approx_ratio"])),
            prog_bar=False,
        )

        ratio_mean, ratio_std = self.compute_mean_std(
            self._test_metrics["approx_ratio"]
        )
        opt_ratio_mean, opt_ratio_std = self.compute_mean_std(
            self._test_metrics["opt_approx_ratio"]
        )
        fixed_angle_ratio_mean, fixed_angle_ratio_std = self.compute_mean_std(
            self._test_metrics["fixed_angle_approx_ratio"]
        )
        reach_mean, reach_std = self.compute_mean_std(self._test_metrics["reach"])
        reach_tail = self.compute_tail_stats(self._test_metrics["reach"])

        self.log("test/approx_ratio_mean", ratio_mean, prog_bar=True,)
        self.log("test/approx_ratio_std", ratio_std, prog_bar=False,)
        self.log(
            "test/opt_approx_ratio_mean", opt_ratio_mean, prog_bar=True,
        )
        self.log(
            "test/opt_approx_ratio_std", opt_ratio_std, prog_bar=False,
        )
        self.log("test/reach_mean", reach_mean, prog_bar=True,)
        self.log("test/reach_std", reach_std, prog_bar=False,)
        self.log("test/reach_min", reach_tail["min"], prog_bar=False, sync_dist=False)
        self.log("test/reach_p5", reach_tail["p5"], prog_bar=False, sync_dist=False)
        self.log("test/reach_p10", reach_tail["p10"], prog_bar=False, sync_dist=False)
        self.log(
            "test/reach_frac_ge_999",
            reach_tail["frac_near"],
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "test/fixed_angle_approx_ratio_mean",
            fixed_angle_ratio_mean,
            prog_bar=True,
        )
        self.log(
            "test/fixed_angle_approx_ratio_std",
            fixed_angle_ratio_std,
            prog_bar=False,
        )

        for i in range(self.depth):
            beta_mean, beta_std = self.compute_mean_std(self._test_params[f"beta_{i}"])
            gamma_mean, gamma_std = self.compute_mean_std(
                self._test_params[f"gamma_{i}"]
            )
            opt_beta_mean, opt_beta_std = self.compute_mean_std(
                self._test_params[f"opt_beta_{i}"]
            )
            opt_gamma_mean, opt_gamma_std = self.compute_mean_std(
                self._test_params[f"opt_gamma_{i}"]
            )
            fixed_angle_beta_mean, fixed_angle_beta_std = self.compute_mean_std(
                self._test_params[f"fixed_angle_beta_{i}"]
            )
            fixed_angle_gamma_mean, fixed_angle_gamma_std = self.compute_mean_std(
                self._test_params[f"fixed_angle_gamma_{i}"]
            )

            self.log(
                f"test/beta_{i + 1}_mean", beta_mean, prog_bar=False,
            )
            self.log(f"test/beta_{i + 1}_std", beta_std, prog_bar=False,)
            self.log(
                f"test/gamma_{i + 1}_mean", gamma_mean, prog_bar=False,
            )
            self.log(
                f"test/gamma_{i + 1}_std", gamma_std, prog_bar=False,
            )
            self.log(
                f"test/opt_beta_{i + 1}_mean",
                opt_beta_mean,
                prog_bar=False,
            )
            self.log(
                f"test/opt_beta_{i + 1}_std",
                opt_beta_std,
                prog_bar=False,
            )
            self.log(
                f"test/opt_gamma_{i + 1}_mean",
                opt_gamma_mean,
                prog_bar=False,
            )
            self.log(
                f"test/opt_gamma_{i + 1}_std",
                opt_gamma_std,
                prog_bar=False,
            )
            self.log(
                f"test/fixed_angle_beta_{i + 1}_mean",
                fixed_angle_beta_mean,
                prog_bar=False,
            )
            self.log(
                f"test/fixed_angle_beta_{i + 1}_std",
                fixed_angle_beta_std,
                prog_bar=False,
            )
            self.log(
                f"test/fixed_angle_gamma_{i + 1}_mean",
                fixed_angle_gamma_mean,
                prog_bar=False,
            )
            self.log(
                f"test/fixed_angle_gamma_{i + 1}_std",
                fixed_angle_gamma_std,
                prog_bar=False,
            )

        self.save_test_per_instance_to_csv()
        self.save_test_metrics_to_csv()
        self._shutdown_eval_pool()

    def save_test_per_instance_to_csv(self) -> None:
        """Dump one row per test instance (reach, approx_ratio, etc.) 
        """
        rows = list(getattr(self, "_test_per_instance", []))
        if getattr(self, "trainer", None) is None or not self.trainer.is_global_zero:
            return

        if not rows:
            self.print("[TEST] No per-instance rows to write.")
            return

        try:
            out_dir = Path(
                getattr(self.trainer, "log_dir", None)
                or self.trainer.default_root_dir
                or "."
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_per_instance.csv"
            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            self.print(f"[TEST] Saved {len(rows)} per-instance rows to {out_path}")
            self.plot_reach_distribution(df, out_dir)
        except Exception as e:
            self.print(f"[WARNING] Failed to save per-instance metrics to CSV: {e}")

    def plot_reach_distribution(self, df: pd.DataFrame, out_dir: Path) -> None:
        """Plot the reach distribution over all evaluated test instances.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            self.print(f"[WARNING] matplotlib unavailable, skipping reach plot: {e}")
            return

        reach = df["reach"].to_numpy(dtype=float)
        instance_sets = sorted(df["instance_set"].unique())
        multi = len(instance_sets) > 1

        fig, axes = plt.subplots(
            1, 2 if multi else 1, figsize=(12 if multi else 6, 4.5), squeeze=False
        )

        ax = axes[0][0]
        ax.hist(reach, bins=50, color="#4C72B0", edgecolor="white")
        ax.axvline(
            reach.mean(),
            color="#C44E52",
            linestyle="--",
            label=f"mean={reach.mean():.4f}",
        )
        ax.axvline(
            np.median(reach),
            color="#55A868",
            linestyle=":",
            label=f"median={np.median(reach):.4f}",
        )
        ax.set_xlabel("reach (model AR / optimal-angle AR)")
        ax.set_ylabel("num instances")
        ax.set_title(f"Reach distribution (n={len(reach)})")
        ax.legend()

        if multi:
            ax2 = axes[0][1]
            for gt in instance_sets:
                vals = df.loc[df["instance_set"] == gt, "reach"].to_numpy(dtype=float)
                ax2.hist(vals, bins=40, alpha=0.5, label=f"{gt} (n={len(vals)})")
            ax2.set_xlabel("reach")
            ax2.set_ylabel("num instances")
            ax2.set_title("Reach by graph type")
            ax2.legend()

        fig.tight_layout()
        plot_path = out_dir / "reach_distribution.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        self.print(f"[TEST] Saved reach distribution plot to {plot_path}")


    def save_test_metrics_to_csv(self) -> None:
        if getattr(self, "trainer", None) is None or not self.trainer.is_global_zero:
            return

        try:
            metrics = dict(self.trainer.callback_metrics)

            rows = []
            for key, value in metrics.items():
                key_str = str(key)
                if not key_str.startswith("test"):
                    continue

                if torch.is_tensor(value):
                    value = value.detach().cpu()
                    if value.numel() != 1:
                        continue
                    value = float(value.item())

                rows.append({"metric": key_str, "value": float(value)})

            if not rows:
                self.print("[TEST] No metrics found to write.")
                return

            out_dir = Path(
                getattr(self.trainer, "log_dir", None)
                or self.trainer.default_root_dir
                or "."
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_metrics.csv"

            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)
            self.print(f"[TEST] Saved {len(df)} metrics to {out_path}")
        except Exception as e:
            self.print(f"[WARNING] Failed to save metrics to CSV: {e}")
