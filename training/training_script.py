# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch/pytorch to avoid segfault
try:
    import juliacall
except ImportError:
    pass

import argparse
import logging
import yaml

from lightning.pytorch.callbacks import ModelCheckpoint
from training.initialization import init_from_config
from training.trainer import build_trainer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def train(cfg: dict) -> None:
    datamodule, model = init_from_config(cfg)
    trainer = build_trainer(cfg)

    trainer.fit(model=model, datamodule=datamodule)

    ckpt_cb = next(cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint))
    logger.info("Model training completed. Best checkpoint saved in: %s", ckpt_cb.best_model_path)


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg=cfg)