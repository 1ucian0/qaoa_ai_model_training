# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Run testing on a single checkpoint file."""

# Import juliacall BEFORE torch to avoid segfault
try:
    import juliacall
except ImportError:
    pass

import argparse
import logging
import pathlib
from pathlib import Path

import torch.serialization
import yaml

from training.initialization import init_from_config
from training.trainer import build_trainer

torch.serialization.add_safe_globals([pathlib.PosixPath])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

for name in ("qiskit", "qiskit.transpiler", "qiskit.compiler"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a single checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to checkpoint file.")
    return parser.parse_args()


def test(cfg: dict, ckpt_path: Path) -> None:
    full_log_path = Path(cfg["trainer"]["log_dir"]) / cfg["trainer"]["run_name"] / "testing"
    full_log_path.mkdir(parents=True, exist_ok=True)
    logger.info("Test logs will be written to: %s", full_log_path)

    datamodule, model = init_from_config(cfg)
    trainer = build_trainer(cfg)

    logger.info("Testing with ckpt_path=%s", ckpt_path)
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(ckpt_path),
    )

    logger.info("Finished testing.")


if __name__ == "__main__":
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    test(cfg=cfg, ckpt_path=args.ckpt_path)