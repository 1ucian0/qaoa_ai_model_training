# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Build results_p<P>/best_param.json from raw qaoa_training_pipeline results.

Reads every result JSON written by `python -m qaoa_training_pipeline.train --save`
under the given folders. For each instance and depth, the candidate with the
highest energy is kept, and one best_param.json is written per depth.

Instances without a file under <graph_dir>/instances/ are skipped. Results not
run with --pre_factor -0.5 (MaxCut) are rejected for the current version of this repo.

Usage:
    python -m scripts_and_notebooks.build_best_param \\
        --results <graph_dir>/results_p1 <graph_dir>/results_p3 \\
        --graphs_repo <graph_dir>
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


def _candidates(result: dict):
    """Yield (qaoa_angles, energy) for every pipeline stage with a numeric energy."""
    for stage in result.values():
        if not isinstance(stage, dict):
            continue
        angles, energy = stage.get("optimized_qaoa_angles"), stage.get("energy")
        # Stages that did not train (e.g. a recursion to the start depth) store "NA"
        if angles and len(angles) % 2 == 0 and isinstance(energy, (int, float)):
            yield angles, float(energy)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="Folders with qaoa_training_pipeline result JSONs, searched recursively.")
    p.add_argument("--graphs_repo", type=Path, required=True)
    p.add_argument("--depths", type=int, nargs="+", default=None,
                   help="QAOA depths to write. Defaults to all depths found.")
    args = p.parse_args()

    repo = args.graphs_repo.resolve()
    instance_dir = repo / "instances"
    if not instance_dir.is_dir():
        sys.exit(f"Error: instance dir not found: {instance_dir}")

    owners = _instance_sets_by_id(instance_dir)

    # best[p][instance_set][instance_id] = {p: entry}
    best: dict[str, dict[str, dict[str, dict]]] = {}
    skipped: set[str] = set()
    for result_file in sorted(f for d in args.results for f in d.rglob("*.json")):
        result = json.loads(result_file.read_text())
        run_args = result.get("args") if isinstance(result, dict) else None
        if not isinstance(run_args, dict) or "input" not in run_args:
            continue  # not a qaoa_training_pipeline result, e.g. best_param.json
        if run_args.get("pre_factor") != -0.5:
            sys.exit(f"Error: {result_file} was not run with --pre_factor -0.5.")
        instance_id = Path(run_args["input"]).stem
        instance_set = owners.get(instance_id)
        if instance_set is None:
            skipped.add(instance_id)
            continue
        for angles, energy in _candidates(result):
            depth = str(len(angles) // 2)
            if args.depths is not None and int(depth) not in args.depths:
                continue
            by_id = best.setdefault(depth, {}).setdefault(instance_set, {})
            current = by_id.get(instance_id)
            if current is None or energy > current[depth]["energy"]:
                by_id[instance_id] = {depth: {
                    "energy": energy,
                    "qaoa_angles": angles,
                    "result_file_name": result_file.name,
                }}

    if not best:
        sys.exit("Error: no results matched the instance files and depths given.")

    for depth in sorted(best, key=int):
        out_file = repo / f"results_p{depth}" / "best_param.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(best[depth], indent=2))
        counts = {s: len(ids) for s, ids in best[depth].items()}
        print(f"p={depth}: {counts} -> {out_file}")

    if skipped:
        print(f"Skipped {len(skipped)} result instances without a file under {instance_dir}")


if __name__ == "__main__":
    main()
