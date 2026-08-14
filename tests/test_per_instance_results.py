# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Guard the per-instance results schema.
"""

import inspect
import pytest

pytest.importorskip("torch")

from training import eval_step

REQUIRED_COLUMNS = [
    "instance_id",
    "instance_set",
    "graph_type",
    "node_count",
    "num_edges",
    "depth_p",
    "reach",
    "approx_ratio",
    "opt_approx_ratio",
    "energy",
    "fixed_angle_approx_ratio",
]


@pytest.fixture(scope="module")
def eval_source() -> str:
    return inspect.getsource(eval_step)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_per_instance_row_includes_column(column, eval_source):
    assert f'"{column}"' in eval_source, (
        f"Column {column!r} is documented in the readme as part of "
        f"test_per_instance.csv but no longer appears in training/eval_step.py"
    )


def test_per_instance_writer_exists():
    assert hasattr(eval_step.QAOAEvalMixin, "save_test_per_instance_to_csv")


def test_predicted_angles_are_recorded(eval_source):
    """beta_<j>/gamma_<j> let the CSV alone reproduce downstream comparisons."""
    assert 'f"beta_{j}"' in eval_source
    assert 'f"gamma_{j}"' in eval_source


def test_instance_set_is_not_a_model_input():
    """The models are designed to stay agnostic to which instance set a graph came from.

    `instance_set` locates files and annotates results. It is a string, so were it
    ever packed into the feature vector, unpack_x would raise on the tensor
    conversion; this test states the intent explicitly.
    """
    from datamodule.meta_configs import SCALAR_FEATURES_IN

    assert "instance_set" not in SCALAR_FEATURES_IN
    assert "graph_type" not in SCALAR_FEATURES_IN
