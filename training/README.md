## Overview of the training code

For setup, configuration and the training and testing commands, see the
[root readme](../README.md). This file describes how the modules in this folder fit
together.

- `training_script` and `testing_script` share the same structure. The training script
  calls `trainer.fit`, while the testing script calls `trainer.test` with a given
  checkpoint path. They are separated because testing can be computationally expensive
  and may be run independently of training (e.g. training on GPU and testing on CPUs).

- `initialization` turns a config dict into the datamodule, model and trainer. It derives
  the conventional `results_p<P>/best_param.json`, `splits/p_<P>/`, `instances/`,
  `max_cuts/` and `fixed_angle_cuts/` paths from `data.graph_dir` and `data.p`, unless
  those are overridden explicitly in the config.

- `eval_step` contains the shared validation and test logic as a mixin for the QAOA
  model. It is placed in a separate file due to its size and extensive logging. It
  implements the energy evaluation by converting edges to graphs and using
  `EfficientDepthOneEvaluator` (p=1) or `PPEvaluator` (p>1, Julia-backed). 

- `qaoa_model` defines the Lightning model wrapper, including the forward pass and
  training step. The current setup uses MSE loss and the Adam optimizer. 

- The trainer setup follows a standard Lightning configuration, including early
  stopping and a TensorBoard logger.
