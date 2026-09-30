# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Build results_p<P>/best_param.json from a QAOA-Parameter-Setting summary.

BestParameterManager in QAOA-Parameter-Setting writes a summary grouped by
evaluator (e.g. "SV", "PP", "MPS") and keyed by instance filename. This
pipeline expects best_param.json grouped by instance set instead. This script
looks up the instance set of every instance in <graph_dir>/instances/<set>/,
keeps only the chosen evaluators, and writes one best_param.json per depth.

Instances without a file under <graph_dir>/instances/ are skipped. If several
evaluators are chosen and have results for the same instance and depth, the
entry with the highest energy is kept.

Usage:
    python -m scripts_and_notebooks.build_best_param \\
        --summary best_parameters_summary.json \\
        --graphs_repo <graph_dir> \\
        --evaluators SV
"""

import argparse
import json
import sys
from pathlib import Path


def _instance_sets_by_id(instance_dir: Path) -> dict[str, str]:
    """Map every instance id under instance_dir/<set>/ to its instance set."""
    owners: dict[str, str] = {}
    for set_dir in sorted(p for p in instance_dir.iterdir() if p.is_dir()):
        for f in set_dir.glob("*.json"):
            if f.stem in owners:
                sys.exit(
                    f"Error: instance id {f.stem} exists in both {owners[f.stem]} and "
                    f"{set_dir.name}. Instance ids must be unique across instance sets."
                )
            owners[f.stem] = set_dir.name
    return owners


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, required=True,
                   help="Summary JSON written by BestParameterManager.save_summary.")
    p.add_argument("--graphs_repo", type=Path, required=True)
    p.add_argument("--evaluators", nargs="+", required=True,
                   help="Top-level summary groups to use, e.g. SV or PP.")
    p.add_argument("--depths", type=int, nargs="+", default=None,
                   help="QAOA depths to write. Defaults to all depths in the summary.")
    args = p.parse_args()

    repo = args.graphs_repo.resolve()
    instance_dir = repo / "instances"
    if not instance_dir.is_dir():
        sys.exit(f"Error: instance dir not found: {instance_dir}")

    summary = json.loads(args.summary.read_text())
    unknown = [e for e in args.evaluators if e not in summary]
    if unknown:
        sys.exit(f"Error: evaluators {unknown} not in summary. Available: {sorted(summary)}")

    owners = _instance_sets_by_id(instance_dir)

    # best[p][instance_set][instance_id] = entry
    best: dict[str, dict[str, dict[str, dict]]] = {}
    skipped: set[str] = set()
    for evaluator in args.evaluators:
        for instance_key, by_depth in summary[evaluator].items():
            instance_id = Path(instance_key).stem
            instance_set = owners.get(instance_id)
            if instance_set is None:
                skipped.add(instance_id)
                continue
            for depth, entry in by_depth.items():
                if args.depths is not None and int(depth) not in args.depths:
                    continue
                by_id = best.setdefault(depth, {}).setdefault(instance_set, {})
                current = by_id.get(instance_id)
                if current is None or entry["energy"] > current[depth]["energy"]:
                    by_id[instance_id] = {depth: entry}

    if not best:
        sys.exit("Error: no summary entries matched the instance files and depths given.")

    for depth in sorted(best, key=int):
        out_file = repo / f"results_p{depth}" / "best_param.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(best[depth], indent=2))
        counts = {s: len(ids) for s, ids in best[depth].items()}
        print(f"p={depth}: {counts} -> {out_file}")

    if skipped:
        print(f"Skipped {len(skipped)} summary instances without a file under {instance_dir}")


if __name__ == "__main__":
    main()
