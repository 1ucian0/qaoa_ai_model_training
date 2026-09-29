# QAOA AI model training

> [!NOTE]
> This repository is associated to a [research publication](<link-to-publication>) and the code here is not actively maintained. This is not an officially supported IBM Quantum software.

This repository provides a training and evaluation pipeline for graph-based machine learning models that predict optimized QAOA parameters directly from a problem graph.

The models take graph-level features and, depending on the architecture, graph topology, node features and edge weights. They predict the QAOA angle vector

```text
[beta_1, ..., beta_p, gamma_1, ..., gamma_p]
```

of length `2p`.

Available model architectures are registered in `MODEL_REGISTRY` in [`ml_models/models.py`](ml_models/models.py). The corresponding model classes in `ml_models/` document which inputs they use.

All commands below are run from the repository root.

## Quickstart

1. Install the package (see [Setup](#setup)).
2. No graph data yet? Generate a tiny synthetic dataset to smoke-test the pipeline:
   ```bash
   python -m scripts_and_notebooks.make_smoke_test_data --out_dir /tmp/qaoa_smoke_test
   python -m scripts_and_notebooks.testset_generation --graphs_repo /tmp/qaoa_smoke_test --instance_sets smoke_test
   python -m scripts_and_notebooks.build_per_p_split --graphs_repo /tmp/qaoa_smoke_test --target_p 1
   python -m training.training_script --config configs/sample_configs/smoke_test_config.yaml
   ```
   This uses random, non-optimized angles purely to check that installation and the pipeline mechanics work end to end. Do not use the resulting checkpoint for anything else.
3. With real data, follow [Data](#data) to lay out `graph_dir`, then [Configuration](#configuration), [Training](#training) and [Testing](#testing) in order.

## Data

**This repository contains code only.** The graph dataset used in the accompanying paper is not included. To train or evaluate a model, you need to provide your own graph data and QAOA optimization results.

The pipeline expects the directory specified by `data.graph_dir` to follow this structure (alternatively, individual paths can also be overwritten manually):

```text
<graph_dir>/
  instances/            # graph instances used as model inputs
  max_cuts/             # optional precomputed max-cut values
  fixed_angle_cuts/     # optional fixed-angle baseline results
  results_p<P>/         # QAOA optimization results for depth P
  splits/               # train, validation and test splits
```

The required data can be generated using the conventions and utilities from [`QAOA-Parameter-Setting`](https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting) together with the `qaoa_training_pipeline` package it depends on.

### 1. Graph instances

Store graph instances under

```text
<graph_dir>/instances/<instance_set>/
```

For example:

```text
<graph_dir>/instances/random_regular/
<graph_dir>/instances/erdos_renyi/
```

Each graph is stored as a JSON edge list with per-edge weights:

```json
{
  "edge list": [
    {"nodes": [0, 1], "weight": 1},
    {"nodes": [1, 2], "weight": 1}
  ]
}
```

See the [`QAOA-Parameter-Setting` instance documentation](https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting/tree/main/instances) for the expected schema and filename conventions.

Every instance ID must be unique across all instance sets that are combined in one dataset.

### 2. Max-cut values

Precomputed maximum and minimum cut values are optional but speed up evaluation.

Store them under

```text
<graph_dir>/max_cuts/<instance_set>/
```

They can be generated with `scripts/compute_mc_for_graph.py` from `QAOA-Parameter-Setting`. For example:

```bash
for f in <graph_dir>/instances/random_regular/*.json; do
  python scripts/compute_mc_for_graph.py "$f" \
    -o "<graph_dir>/max_cuts/random_regular/$(basename "$f")"
done
```

If these files are not available, `training/eval_step.py` computes the required values during evaluation.

### 3. QAOA optimization results

The supervised labels are optimized QAOA parameters for each graph instance and QAOA depth.

Store the raw optimization results under

```text
<graph_dir>/results_p<P>/<instance_set>/
```

For example:

```text
<graph_dir>/
└── results_p3/
    ├── random_regular/
    │   ├── <result>.json
    │   └── ...
    └── erdos_renyi/
        ├── <result>.json
        └── ...
```

You can use any QAOA parameter optimization strategy that produces the required results. [`QAOA-Parameter-Setting`](https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting) provides example strategies and utilities for generating them.

For training, each `results_p<P>/` directory also needs a `best_param.json` containing the best angle and energy pair for every instance:

```text
{
  <instance_set>: {
    <instance_id>: {
      <p>: {
        "energy": ...,
        "qaoa_angles": ...,
        "result_file_name": ...
      }
    }
  }
}
```

`QAOA-Parameter-Setting` provides `BestParameterManager` for aggregating optimization results. 

### 4. Train, validation and test splits

Once the graph instances are available, generate the master split with

```bash
python -m scripts_and_notebooks.testset_generation \
  --graphs_repo <graph_dir> \
  --instance_sets random_regular erdos_renyi
```

This also generates the normalization statistics used by the models.

For a specific QAOA depth, filter the split to instances that have optimization results at that depth:

```bash
python -m scripts_and_notebooks.build_per_p_split \
  --graphs_repo <graph_dir> \
  --target_p 3
```

This creates

```text
splits/p_<P>/train_val_split.json
splits/p_<P>/test_split.json
```

Point `data.split_file` and `data.test_file` in the configuration to these files.

### Fixed-angle baseline

`fixed_angle_cuts/` is optional. If no fixed-angle result is available for an instance, training and testing still work and `fixed_angle_approx_ratio` is left empty.

To reproduce our fixed-angle baseline:

1. Download `angles_regular_graphs.json` from [danlkv/fixed-angle-QAOA at commit `956be14`](https://github.com/danlkv/fixed-angle-QAOA/blob/956be1450c9d1f1957d3189a6603264d13163088/angles_regular_graphs.json).
2. For each instance set, run:
   ```bash
   python -m scripts_and_notebooks.compute_fixed_angle_cuts \
     --angles_file angles_regular_graphs.json \
     --instance_dir <graph_dir>/instances/<instance_set> \
     --instance_set <instance_set> \
     --depth <P>
   ```
   This looks up the published angles for each instance's degree, evaluates the cut value, and writes one result file per instance under `fixed_angle_cuts/p_<P>/<instance_set>/`.

You can use the same output format to plug in fixed angles from any other source instead.

A fixed-angle result has the following format:

```json
{
  "fixed_angle_cut": 7.634962789623944,
  "betas": [0.3926821085699073],
  "gammas": [1.1598139097947238]
}
```

`fixed_angle_cut` is the cut value at the given angles. `betas` and `gammas` must each have length `p`.

## Setup

Requires Python 3.10+.

```bash
pip install -e .
```

This covers the `mlp`, `edge_transformer`, `diffusion_transformer`, and `graph_neural_network` models.

The `gcn`, `graph_isomorphism_network`, and `graph_transformer` models need `torch-geometric` as well. Install it after the base package, once `torch` is already present:

```bash
python -m pip install -e ".[graph]" --no-build-isolation
```

`--no-build-isolation` stops `torch-scatter` from building against a fresh, torch-less environment instead of the one you just installed `torch` into.


### Julia for QAOA depth greater than 1

Evaluation at `p = 1` uses `EfficientDepthOneEvaluator` and only requires Python.

For `p > 1`, the pipeline uses `PPEvaluator`, which relies on [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl).

Install Julia first, then run:

```bash
pip install -e ".[julia]"
python link-julia-python.py
```

The setup script installs `PauliPropagation.jl` into the Julia environment.

`juliacall` must be imported before `torch` to avoid a segfault. The package handles this import order internally. Dataloader workers use the spawn start method because `juliacall` is not fork-safe.

## Configuration

[`configs/sample_configs/sample_config.yaml`](configs/sample_configs/sample_config.yaml) is the annotated reference configuration. Copy it and adapt it for your experiment.

The main settings to change are:

- **`data.graph_dir`**: root directory containing the graph data
- **`data.p`**: QAOA depth. Start with `p = 1` if you want to run the pipeline without Julia
- **`data.instance_set`**: instance sets used for training
- **`data.norm_stats`**: feature normalization statistics. Providing them explicitly gives reproducible normalization
- **`model.model_type`**: model architecture from `MODEL_REGISTRY`

Multiple instance sets can be listed to train one model across different graph families.

[`scripts_and_notebooks/datamodule_tutorial.ipynb`](scripts_and_notebooks/datamodule_tutorial.ipynb) provides a walkthrough of the datamodule, normalization statistics and data splits.

## Training

Run training with:

```bash
python -m training.training_script \
  --config configs/sample_configs/sample_config.yaml
```

Checkpoints are written to

```text
<ckpt_dir>/<run_name>/<model_type>-p<p>-<YYYYMMDD_HHMMSS>-<epoch>-<metric>.ckpt
```

TensorBoard logs are written to

```text
<log_dir>/<run_name>/
```

Both locations are configured in the `trainer` section of the configuration file.

## Testing

Testing is run separately from training because QAOA evaluation can be expensive and may be performed on different hardware.

Test a checkpoint with:

```bash
python -m training.testing_script \
  --config configs/sample_configs/sample_config.yaml \
  --ckpt_path checkpoints/p_1/rr/last.ckpt
```

To evaluate several checkpoints, run `training.testing_script` once for each configuration and checkpoint pair.

Each run writes

```text
test_metrics.csv
test_per_instance.csv
```

to its log directory.

### Analysing results

`test_per_instance.csv` contains one row for each evaluated graph and is the recommended basis for further analysis.

| Column | Meaning |
|---|---|
| `instance_id`, `instance_set` | Graph instance and source instance set |
| `node_count`, `num_edges`, `depth_p` | Graph size and QAOA depth |
| `reach` | Model approximation ratio divided by the optimal-angle approximation ratio |
| `approx_ratio` | Approximation ratio obtained from the predicted angles |
| `opt_approx_ratio` | Approximation ratio obtained from the reference angles |
| `energy` | QAOA energy obtained from the predicted angles |
| `fixed_angle_approx_ratio` | Fixed-angle baseline when available |
| `beta_<j>`, `gamma_<j>` | Predicted QAOA angles |

A `reach` of `1.0` means that the predicted angles match the approximation ratio obtained from the reference optimized angles.

For example:

```python
import pandas as pd

df = pd.read_csv("runs/<run_name>/test_per_instance.csv")

df.groupby("instance_set")["reach"].agg(["mean", "std", "min"])
df.groupby("node_count")["reach"].mean()
(df["reach"] > 0.99).mean()
df.nsmallest(10, "reach")
```

The models do not receive `instance_set` or `graph_type` as input. These fields are only stored for analysis.

Checkpoints are loaded with `strict=True`, so incompatible model architectures fail during loading instead of silently loading a partially initialized model.

## Repository layout

| Path | Contents |
|---|---|
| `datamodule/` | Dataset loading, feature computation, normalization and splits |
| `ml_models/` | Model architectures and `MODEL_REGISTRY` |
| `training/` | Training, testing and QAOA evaluation |
| `configs/sample_configs/` | Annotated example configuration |
| `scripts_and_notebooks/` | Data preparation utilities and tutorial |
| `tests/` | Model, configuration and repository tests |

See [`training/README.md`](training/README.md) for how the modules in `training/` fit together.

The main training components are:

- `training/training_script.py` runs model training
- `training/testing_script.py` evaluates a trained checkpoint
- `training/eval_step.py` contains the shared validation and test evaluation logic
- `training/qaoa_model.py` defines the Lightning model, training step, MSE loss and Adam optimizer

The data preparation utilities are:

| File | Purpose |
|---|---|
| `datamodule_tutorial.ipynb` | Walkthrough of the datamodule, normalization and splits |
| `testset_generation.py` | Generate the master train, validation and test split |
| `build_per_p_split.py` | Filter a split to instances available at a given QAOA depth |


## Acknowledgements

This project builds on [`qaoa_training_pipeline`](https://github.com/qiskit-community/qaoa_training_pipeline) for QAOA evaluation and training utilities and on [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl) for efficient evaluation at QAOA depth `p > 1`.

## License

See [LICENSE](LICENSE).