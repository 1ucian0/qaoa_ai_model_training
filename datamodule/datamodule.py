# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch to avoid segfault warning
# See: https://github.com/pytorch/pytorch/issues/78829
try:
    import juliacall
except ImportError:
    pass  # juliacall is optional

import json
import random
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from lightning.pytorch import LightningDataModule

from datamodule.dataset import QAOADataset
from datamodule.collate import qaoa_collate_fn
from datamodule.meta_configs import BASIC_IN_FEATURES, VALID_OUT_FEATURES

logger = logging.getLogger("[QAOA Data Module]")


class QAOADataModule(LightningDataModule):
    def __init__(
        self,
        summary_file: str | Path,
        instance_dir: str | Path,
        target_p: int,
        cut_metrics_dir: str | Path | None = None,
        instance_set: list[str] = ["random_regular"],
        in_features: list[str] | None = BASIC_IN_FEATURES,
        out_features: list[str] | None = VALID_OUT_FEATURES,
        num_timesteps: int = 0,
        compute_cut_metrics: bool = False,
        use_edge_weights: bool = False,
        normalize: bool = True,
        norm_stats: dict | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_fraction: float = 0.8,
        train_val_split_seed: int = 42,
        shuffle_train: bool = True,
        split_file: str | Path = None,
        save_split: bool = False,
        test_file: str | Path = None,
    ) -> None:
        """
        Args:
        summary_file: Path to summary file.
        instance_dir: Path to instance directory.
        cut_metrics_dir: Path to cut metrics directory.
        instance_set: Instance directories to load from.
        in_features: List of input features to return in get_item.
        out_features: List of output features to return in get_item.
        target_p: Which QAOA depth p to load.
        compute_cut_metrics: Whether to compute max/min cut metrics.
        normalize(bool): Whether to normalize input features.
        norm_stats (dict): Normalization statistics.
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of workers for dataloader
        pin_memory (bool): Whether to pin memory
        train_fraction (float): Fraction of training data
        train_val_split_seed (int): Random seed
        shuffle_train (bool): Whether to shuffle training data
        """

        super().__init__()
        self.summary_file = Path(summary_file)
        self.instance_dir = Path(instance_dir)
        self.cut_metrics_dir = Path(cut_metrics_dir) if cut_metrics_dir else None
        self.instance_set = instance_set
        self.in_features = list(in_features) if in_features is not None else []
        self.out_features = list(out_features) if out_features is not None else []
        self.target_p = target_p
        self.compute_cut_metrics = compute_cut_metrics
        self.use_edge_weights = bool(use_edge_weights)
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.num_timesteps = num_timesteps

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = self.num_workers > 0

        self.train_fraction = float(train_fraction)
        if not (0.0 <= train_fraction <= 1.0):
            raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

        self.train_val_split_seed = int(train_val_split_seed)
        self.shuffle_train = bool(shuffle_train)

        self.split_file = Path(split_file) if split_file is not None else None
        self.save_split = bool(save_split)
        self.test_file = Path(test_file) if test_file is not None else None

        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None
        self.test_dataset: Subset | None = None

    def _split_meta(self, dataset_len: int) -> dict[str, any]:
        return {
            "dataset_len": dataset_len,
            "train_fraction": self.train_fraction,
            "seed": self.train_val_split_seed,
            "summary_file": str(self.summary_file.expanduser().resolve()),
            "instance_dir": str(self.instance_dir.expanduser().resolve()),
            "instance_set": list(self.instance_set),
        }

    def save_split_indices(
        self,
        path: Path,
        train_instance_ids: list[int],
        val_instance_ids: list[int],
        meta: dict[str, any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "train_indices": train_instance_ids,
            "val_indices": val_instance_ids,
            "meta": meta,
        }
        path.write_text(json.dumps(info, indent=2, sort_keys=False))

    def load_split_indices(
        self, path: Path
    ) -> tuple[list[int], list[int], dict[str, any]]:
        if not path.exists():
            raise FileNotFoundError(path)

        info = json.loads(path.read_text())
        train_instance_ids = info.get("train_indices")
        val_instance_ids = info.get("val_indices")
        meta = info.get("meta", {})

        if not isinstance(train_instance_ids, list) or not all(
            isinstance(i, str) for i in train_instance_ids
        ):
            raise ValueError("split_file train_indices must be a list[str]")
        if not isinstance(val_instance_ids, list) or not all(
            isinstance(i, str) for i in val_instance_ids
        ):
            raise ValueError("split_file val_indices must be a list[str]")

        if len(set(train_instance_ids)) != len(train_instance_ids):
            raise ValueError("split_file train_indices contains duplicates")
        if len(set(val_instance_ids)) != len(val_instance_ids):
            raise ValueError("split_file val_indices contains duplicates")
        if set(train_instance_ids) & set(val_instance_ids):
            raise ValueError("split_file train_indices and val_indices overlap")

        return train_instance_ids, val_instance_ids, meta

    def load_test_indices(self, path: Path) -> tuple[list[int], dict[str, any]]:
        if not path.exists():
            raise FileNotFoundError(path)

        info = json.loads(path.read_text())
        test_instance_ids = info.get("test_indices")
        meta = info.get("meta", {})

        if len(set(test_instance_ids)) != len(test_instance_ids):
            raise ValueError("test_file test_indices contains duplicates")

        return test_instance_ids, meta

    def _make_split_indices(
        self,
        instance_ids: list[str],
        excluded_ids: set[str],
    ) -> tuple[list[str], list[str]]:
        """Split the remaining instances into train and validation sets.

        A uniform random split, seeded by ``train_val_split_seed`` so it is
        reproducible. Instance names are treated as identifiers.
        """
        pool = [iid for iid in instance_ids if iid not in excluded_ids]
        n = len(pool)
        if n == 0:
            raise ValueError(
                "No instances left for train/val after excluding test set."
            )

        if self.train_fraction == 0.0:
            n_train = 0
        elif self.train_fraction == 1.0:
            n_train = n
        else:
            n_train = int(round(self.train_fraction * n))
            n_train = max(1, min(n_train, n - 1))
        n_val = n - n_train

        if n_val == 0:
            return pool, []
        if n_train == 0:
            return [], pool

        shuffled = sorted(pool)
        random.Random(self.train_val_split_seed).shuffle(shuffled)
        val_set = set(shuffled[:n_val])

        # Preserve the original ordering within each split.
        val_ids = [iid for iid in pool if iid in val_set]
        train_ids = [iid for iid in pool if iid not in val_set]
        return train_ids, val_ids

    def setup(self, stage: str | None = None) -> None:
        logger.info("Setting up datamodule.")


        ds = QAOADataset(
            summary_file=self.summary_file,
            instance_dir=self.instance_dir,
            cut_metrics_dir=self.cut_metrics_dir,
            instance_set=self.instance_set,
            in_features=self.in_features,
            out_features=self.out_features,
            target_p=self.target_p,
            compute_cut_metrics=self.compute_cut_metrics,
            num_timesteps=self.num_timesteps,
            use_edge_weights=self.use_edge_weights,
        )

        id_to_idx = {}

        for i, s in enumerate(ds.samples):
            iid = s.get("instance_id")
            if isinstance(iid, str) and iid not in id_to_idx:
                id_to_idx[iid] = i

        all_ids = list(id_to_idx.keys())
        test_ids = []

        def _resolve(iid):
            return id_to_idx.get(iid)

        if self.test_file is not None and self.test_file.exists():
            test_ids, _meta = self.load_test_indices(self.test_file)
        excluded = set(test_ids)

        if self.split_file is not None and self.split_file.exists():
            train_ids, val_ids, meta = self.load_split_indices(self.split_file)

            if set(train_ids) & excluded or set(val_ids) & excluded:
                raise ValueError(
                    "split_file train/val overlap with test_file hold-out set."
                )

            self.train_dataset = Subset(
                ds, [j for iid in train_ids if (j := _resolve(iid)) is not None]
            )
            self.val_dataset = Subset(
                ds, [j for iid in val_ids if (j := _resolve(iid)) is not None]
            )

            missing_train = sorted(iid for iid in train_ids if _resolve(iid) is None)
            missing_val = sorted(iid for iid in val_ids if _resolve(iid) is None)

            if missing_train or missing_val:
                logger.warning(
                    "split_file contains unknown instance_ids "
                    f"(train_missing={len(missing_train)}, val_missing={len(missing_val)}). "
                    "Check if this is NOT due to purposeful subset training (e.g., a limited instance_set)."
                )

            logger.info(
                f"Loaded split from {self.split_file} "
                f"(train={len(train_ids) - len(missing_train)}, val={len(val_ids) - len(missing_val)})."
            )

        else:
            train_ids, val_ids = self._make_split_indices(
                all_ids, excluded_ids=excluded
            )
            self.train_dataset = Subset(ds, [id_to_idx[iid] for iid in train_ids])
            self.val_dataset = Subset(ds, [id_to_idx[iid] for iid in val_ids])

            if self.split_file is not None and self.save_split:
                meta = self._split_meta(dataset_len=len(ds))
                self.save_split_indices(self.split_file, train_ids, val_ids, meta)
                logger.info(
                    f"Saved split to {self.split_file} (train={len(train_ids)}, val={len(val_ids)})."
                )
        if test_ids:
            missing_test = sorted(iid for iid in test_ids if _resolve(iid) is None)
            if missing_test:
                logger.warning(
                    "test_file contains unknown instance_ids "
                    f"(test_missing={len(missing_test)}). "
                    "Check if this is NOT due to purposeful subset testing."
                )

            test_idx = [j for iid in test_ids if (j := _resolve(iid)) is not None]

            logger.info(
                f"Loaded test split from {self.test_file} "
                f"(test={len(test_idx)})."
            )

            self.test_dataset = Subset(ds, test_idx)

        if self.normalize:
            if self.norm_stats is not None:
                ds.set_normalizer_stats(self.norm_stats)
                logger.info("Using precomputed normalization statistics.")

            else:
                logger.info(
                    "Normalize is set to True; computing normalization statistics on training data, which may take some time."
                )
                train_idx = self.train_dataset.indices
                stats = ds.compute_normalizer_stats(train_idx)
                ds.set_normalizer_stats(stats)
                self.norm_stats = stats

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=qaoa_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=qaoa_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test set configured.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=qaoa_collate_fn,
        )
