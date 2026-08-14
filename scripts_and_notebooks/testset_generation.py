# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Generate reusable train/val/test split files over a corpus of graph instances.

Splits are anchored to the instance files available, which ensures robustness when generating split files
across QAOA depths with varying availability of solved angles, e.g. an instance is
held out regardless of which depths have results for it for comparable results across depths.

Instance names are treated as identifiers, and splits are uniform random with a fixed seed.
"""

import json
import logging
import random
import argparse
from pathlib import Path

import numpy as np

logger = logging.getLogger("[QAOA Test Set Generation]")


def _master_ids(instance_dir: Path, instance_set: str) -> list[str]:
    """Every instance id in one instance directory, sorted for determinism."""
    set_dir = instance_dir / instance_set
    if not set_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {set_dir}")

    return sorted(p.stem for p in set_dir.glob("*.json"))


def _gather_by_set(instance_dir: Path, instance_sets: list[str]) -> dict[str, list[str]]:
    by_set = {name: _master_ids(instance_dir, name) for name in instance_sets}

    # Split files store ids without their instance set, so ids must be globally unique.
    owners: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for set_name, ids in by_set.items():
        for iid in ids:
            if iid in owners:
                duplicates.append((iid, owners[iid], set_name))
            else:
                owners[iid] = set_name

    if duplicates:
        examples = ", ".join(
            f"{iid} ({first}, {second})"
            for iid, first, second in duplicates[:5]
        )
        raise ValueError(
            "Instance ids must be unique across instance sets. "
            f"Duplicate ids found: {examples}"
        )

    return by_set


def _split_ids(ids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split ``ids`` into ``(selected, remaining)``, selecting ``fraction`` of them.

    Applied per instance set so every set contributes proportionally, and seeded
    so the result is reproducible.
    """
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n_select = max(0, min(int(round(fraction * len(shuffled))), len(shuffled)))
    selected = set(shuffled[:n_select])
    return (
        [i for i in sorted(ids) if i in selected],
        [i for i in sorted(ids) if i not in selected],
    )


def _build_meta(
    instance_sets: list[str],
    instance_dir: Path,
    test_fraction: float,
    train_fraction: float,
    seed: int,
    by_set: dict[str, list[str]],
) -> dict:
    return {
        "instance_dir": str(instance_dir),
        "instance_sets": instance_sets,
        "test_fraction": test_fraction,
        "train_fraction": train_fraction,
        "seed": seed,
        "counts_by_instance_set": {k: len(v) for k, v in by_set.items()},
    }


def _graph_structural_features(graph_path: Path) -> dict:
    with graph_path.open() as f:
        raw = json.load(f)

    edges = raw["edge list"]
    nodes: set = set()
    degree: dict = {}

    for e in edges:
        u, v = e["nodes"]
        nodes.add(u)
        nodes.add(v)
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    n_nodes = len(nodes)
    n_edges = len(edges)
    degrees = list(degree.values()) or [0]

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "edges_per_node": n_edges / max(n_nodes, 1),
        "mean_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees)),
    }


def _structural_norm_stats(
    instance_dir: Path, train_ids_with_set: list[tuple[str, str]]
) -> dict:
    """Mean/std of the basic graph features over the training instances."""
    if not train_ids_with_set:
        raise ValueError(
            "Cannot compute normalization statistics without training instances."
        )

    feats = ["num_nodes", "num_edges", "edges_per_node", "mean_degree", "std_degree"]
    vals: dict[str, list[float]] = {f: [] for f in feats}

    for iid, set_name in train_ids_with_set:
        feat = _graph_structural_features(
            instance_dir / set_name / f"{iid}.json"
        )
        for f in feats:
            vals[f].append(feat[f])

    out: dict = {"x": {}}
    for f, vs in vals.items():
        arr = np.asarray(vs, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        out["x"][f] = {
            "mean": mean,
            "std": std if std > 1e-12 else 1.0,
        }

    return out


def generate_test_indices_file(
    instance_dir: str | Path,
    out_path: str | Path,
    instance_set: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], Path]:
    """Write a master test split file. Returns ``(test_ids, out_path)``.

    Generate this once and reuse it so the same instances stay held out across
    every run you intend to compare. Regenerating after adding or removing
    instances from the master corpus may change split membership.
    """
    tf = float(test_fraction)
    if not (0.0 < tf < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {tf!r}")

    instance_dir = Path(instance_dir).expanduser().resolve()
    instance_sets = list(instance_set)
    by_set = _gather_by_set(instance_dir, instance_sets)

    test_ids: list[str] = []
    for ids in by_set.values():
        selected, _ = _split_ids(ids, tf, seed)
        test_ids.extend(selected)
    test_ids.sort()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = _build_meta(instance_sets, instance_dir, tf, 0.0, seed, by_set)

    out_path.write_text(
        json.dumps(
            {"test_indices": test_ids, "meta": meta},
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
    )

    logger.info("Wrote test split: %d ids -> %s", len(test_ids), out_path)
    return test_ids, out_path


def generate_split_indices_file(
    instance_dir: str | Path,
    out_path: str | Path,
    instance_set: list[str],
    test_fraction: float = 0.2,
    train_fraction: float = 0.9,
    seed: int = 42,
    compute_norm_stats: bool = True,
) -> tuple[list[str], list[str], Path]:
    """Write a master train/val split file with optional embedded ``norm_stats``.

    ``train_fraction`` is the fraction of non-test instances assigned to
    training. For example, test_fraction=0.2 and train_fraction=0.9 give
    approximately 72% train, 8% validation, and 20% test overall.

    Pass the same ``test_fraction`` and ``seed`` as
    :func:`generate_test_indices_file` so the held-out instances are excluded
    consistently.
    """
    tf = float(test_fraction)
    trf = float(train_fraction)

    if not (0.0 < tf < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {tf!r}")
    if not (0.0 < trf <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {trf!r}")

    instance_dir = Path(instance_dir).expanduser().resolve()
    instance_sets = list(instance_set)
    by_set = _gather_by_set(instance_dir, instance_sets)

    train_ids: list[str] = []
    val_ids: list[str] = []
    set_lookup: dict[str, str] = {}

    for name, ids in by_set.items():
        _, remaining = _split_ids(ids, tf, seed)

        # Offset the seed so train/val selection is independent of test selection.
        set_train, set_val = _split_ids(remaining, trf, seed + 1)

        train_ids.extend(set_train)
        val_ids.extend(set_val)
        set_lookup.update({iid: name for iid in ids})

    train_ids.sort()
    val_ids.sort()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = _build_meta(instance_sets, instance_dir, tf, trf, seed, by_set)
    payload: dict = {
        "train_indices": train_ids,
        "val_indices": val_ids,
        "meta": meta,
    }

    if compute_norm_stats:
        logger.info(
            "Computing norm_stats over %d training instances...",
            len(train_ids),
        )
        payload["norm_stats"] = _structural_norm_stats(
            instance_dir,
            [(iid, set_lookup[iid]) for iid in train_ids],
        )

    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
    )

    logger.info(
        "Wrote train/val split: train=%d, val=%d -> %s",
        len(train_ids),
        len(val_ids),
        out_path,
    )

    return train_ids, val_ids, out_path

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs_repo", type=Path, required=True)
    p.add_argument("--instance_sets", nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Defaults to <graphs_repo>/splits/.")
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--train_fraction", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    instance_dir = args.graphs_repo.resolve() / "instances"
    if not instance_dir.is_dir():
        sys.exit(f"Error: instance dir not found: {instance_dir}")

    out_dir = args.out_dir or (args.graphs_repo.resolve() / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating master splits for {args.instance_sets} -> {out_dir}")
    generate_test_indices_file(
        instance_dir=instance_dir,
        out_path=out_dir / "test_split.json",
        instance_set=args.instance_sets,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    generate_split_indices_file(
        instance_dir=instance_dir,
        out_path=out_dir / "train_val_split.json",
        instance_set=args.instance_sets,
        test_fraction=args.test_fraction,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    print("Done.")

if __name__ == "__main__":
    main()