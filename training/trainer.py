# Copyright IBM Corp. 2026
# License: Apache-2.0

import os
from datetime import datetime
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger


def build_trainer(cfg: dict) -> pl.Trainer:
    trainer_cfg = cfg["trainer"]
    model_cfg = cfg["model"]

    os.makedirs(trainer_cfg["log_dir"], exist_ok=True)
    os.makedirs(trainer_cfg["ckpt_dir"], exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set checkpoint and log directories to run_name folder
    run_name = trainer_cfg.get("run_name", "param_prediction")
    ckpt_dir = os.path.join(trainer_cfg["ckpt_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor = trainer_cfg.get("monitor", "val/loss")
    mode = trainer_cfg.get("monitor_mode", "min")

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=(
            f"{model_cfg['model_type']}"
            f"-p{cfg['data']['p']}"
            f"-{run_id}" + "-{epoch:02d}-{" + monitor + ":.4f}"
        ),
        monitor=monitor,
        mode=mode,
        save_top_k=int(trainer_cfg.get("save_top_k", 1)),
        save_last=bool(trainer_cfg.get("save_last", False)),
        auto_insert_metric_name=False,
    )

    es_cb = pl.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=int(model_cfg.get("patience", 10)),
        min_delta=float(model_cfg.get("min_delta", 0.0)),
    )

    # Logs will be saved directly in log_dir/run_name/ and overwrite previous runs
    logger = TensorBoardLogger(
        save_dir=trainer_cfg["log_dir"],
        name=run_name,
        version="",
    )

    return pl.Trainer(
        max_epochs=int(model_cfg["num_epochs"]),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        strategy=trainer_cfg.get("strategy", "auto"),
        default_root_dir=trainer_cfg["ckpt_dir"],
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        # "warn" uses deterministic algorithms where available and only warns for ops
        # without one, e.g. scatter-add on Apple MPS, instead of failing.
        deterministic=trainer_cfg.get("deterministic", "warn"),
        check_val_every_n_epoch=int(trainer_cfg.get("check_val_every_n_epoch", 5)),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
