# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Validate sample YAML training configs.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_DIR = PROJECT_ROOT / "configs" / "sample_configs"

SAMPLE_CONFIGS = sorted(SAMPLE_CONFIG_DIR.glob("*.yaml"))

# Names the training entry point can build. Imported lazily so the YAML checks
# still run when torch is unavailable.
def _training_model_names() -> set[str]:
    from ml_models.models import MODEL_REGISTRY

    return set(MODEL_REGISTRY)


def test_sample_configs_exist():
    assert SAMPLE_CONFIGS, f"No sample configs found under {SAMPLE_CONFIG_DIR}"


@pytest.mark.parametrize("config_path", SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_sample_config_parses(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    assert isinstance(cfg, dict), f"{config_path.name}: not a mapping"


@pytest.mark.parametrize("config_path", SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_sample_config_has_required_keys(config_path):
    """Mirror what training.initialization.init_from_config requires up front.
    """
    cfg = yaml.safe_load(config_path.read_text())

    assert "data" in cfg, f"{config_path.name}: missing top-level 'data'"
    assert "model" in cfg, f"{config_path.name}: missing top-level 'model'"

    data = cfg["data"]
    assert "graph_dir" in data, (
        f"{config_path.name}: missing data.graph_dir — init_from_config reads this "
        f"unconditionally and would raise KeyError"
    )
    assert "p" in data, f"{config_path.name}: missing data.p"

    model = cfg["model"]
    assert "model_type" in model, f"{config_path.name}: missing model.model_type"
    assert "num_timesteps" in model, (
        f"{config_path.name}: missing model.num_timesteps — read unconditionally "
        f"by init_from_config"
    )


@pytest.mark.parametrize("config_path", SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_sample_config_model_type_is_buildable(config_path):
    pytest.importorskip("torch")
    cfg = yaml.safe_load(config_path.read_text())
    model_type = str(cfg["model"]["model_type"]).lower()
    known = _training_model_names()
    assert model_type in known, (
        f"{config_path.name}: model_type {model_type!r} is not in the training "
        f"registry; get_model would raise ValueError. Known: {sorted(known)}"
    )