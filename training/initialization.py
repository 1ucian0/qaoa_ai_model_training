# Copyright IBM Corp. 2026
# License: Apache-2.0

import logging
import lightning.pytorch as pl
import json
from training.qaoa_model import QAOAModel
from datamodule.datamodule import QAOADataModule
from datamodule.meta_configs import NODE_FEATURES, EDGE_FEATURES
from pathlib import Path
from datamodule.utils import set_seed

logger = logging.getLogger(__name__)

# Always requested from the datamodule: metadata and raw-graph fields the models
# and evaluation need, regardless of what the config lists as in_features.
IN_FEATURES_ADD = {
    "node_count", "edges", "edge_weights", "instance_id", "instance_set",
    "t", "rescale_a",
}
OUT_FEATURES_ADD = {"qaoa_params", "opt_energy", "opt_max_cut", "opt_min_cut", "opt_approx_ratio"}

def init_from_config(cfg: dict) -> tuple:

    logger.info("Starting initialization from config...")
    set_seed(int(cfg.get("seed", 42)))

    data = cfg["data"]
    num_timesteps = cfg["model"]["num_timesteps"]

    graph_dir = Path(data["graph_dir"]).expanduser().resolve()
    p = int(data["p"])
    logger.info("Graph dir: %s", graph_dir)
    logger.info("Problem depth p: %d", p)

    # Default dir conventions
    results_dir = graph_dir / f"results_p{p}"
    instance_dir = graph_dir / "instances"
    cut_metrics_dir = graph_dir / "max_cuts"
    fixed_angle_dir = graph_dir / "fixed_angle_cuts"

    logger.info("Results dir: %s", results_dir)

    if "instance_set" in data:
        instance_sets = data["instance_set"]
    elif "graph_type" in data:
        instance_sets = data["graph_type"]
        logger.warning(
            "Config key 'graph_type' is deprecated; rename it to 'instance_set'."
        )
    else:
        instance_sets = ["graphs"]

    if isinstance(instance_sets, str):
        instance_sets = [instance_sets]

    logger.info("Instance sets: %s", instance_sets)

    default_summary = results_dir / "best_param.json"
    default_split   = results_dir / "train_val_split.json"
    default_test    = results_dir / "test_split.json"

    # Allow overrides from config, otherwise use defaults
    summary_file = Path(data.get("summary_file", default_summary))
    split_file   = Path(data.get("split_file", default_split))
    test_file    = Path(data.get("test_file", default_test))
    instance_dir = Path(data.get("instance_dir", instance_dir))
    cut_metrics_dir = Path(data.get("cut_metrics_dir", cut_metrics_dir))
    fixed_angle_dir = Path(data.get("fixed_angle_dir", fixed_angle_dir))

    logger.info("Summary file: %s", summary_file)
    logger.info("Split file: %s", split_file)
    logger.info("Test file: %s", test_file)

    if bool(data.get("recompute_norm_stats", False)):
        norm_stats = None
        logger.info("recompute_norm_stats=True; will compute norm_stats on train split")
    elif "norm_stats" in data and data["norm_stats"] is not None:
        norm_stats = data["norm_stats"]
        logger.info("Using norm_stats from config")
    else:
        if split_file.exists():
            logger.info("Loading norm_stats from split file...")
            with open(split_file, "r") as f:
                js = json.load(f)
                norm_stats = js.get("norm_stats", None)
        else:
            logger.info("No norm_stats found")
            norm_stats = None

    use_edge_weights = bool(data.get("use_edge_weights", True))

    dm = QAOADataModule(
        summary_file=summary_file,
        instance_dir=instance_dir,
        cut_metrics_dir=cut_metrics_dir,
        target_p=p,
        instance_set=instance_sets,
        in_features=list(set(data["in_features"]) | IN_FEATURES_ADD),
        out_features=list(OUT_FEATURES_ADD),
        num_timesteps=num_timesteps,
        norm_stats=norm_stats,
        batch_size=int(data.get("batch_size", 8)),
        num_workers=int(data.get("num_workers", 0)),
        split_file=split_file,
        test_file=test_file,
        use_edge_weights=use_edge_weights,
    )

    logger.info("DataModule initialized")

    model_cfg = cfg["model"]
    p = int(data["p"])
    output_dim = 2 * p

    target_key = model_cfg.get("target_key", "qaoa_params")

    in_feats = list(data["in_features"])
    non_scalar = set(NODE_FEATURES) | set(EDGE_FEATURES) | {"edges", "edge_weights"}
    scalar_feats = [f for f in in_feats if f not in non_scalar]
    input_dim = len(scalar_feats)

    logger.info("Model type: %s", model_cfg["model_type"])

    model_type = model_cfg["model_type"]
    if "diffusion" in model_type and not num_timesteps:
        raise ValueError("diffusion model requires positive number of denoising timesteps")
    elif "diffusion" not in model_type and num_timesteps:
        logger.warning("Ignoring num_timesteps, parameter is specific to diffusion models. Currently using %s.", model_type)
        num_timesteps = 0

    model = QAOAModel(
        model_type=model_type,
        in_features=in_feats,
        target_key=target_key,
        input_dim=input_dim,
        output_dim=output_dim,
        lr=float(model_cfg["lr"]),
        depth=p,
        timesteps=num_timesteps,
        instance_dir=instance_dir,
        fixed_angle_dir=fixed_angle_dir,
        use_degree_flag=bool(model_cfg.get("use_degree_flag", False)),
        use_triangle_features=bool(model_cfg.get("use_triangle_features", False)),
        summary_file=str(summary_file),
        test_eval_workers=int(model_cfg.get("test_eval_workers", 1)),
    )

    logger.info("Model initialized successfully")

    return dm, model