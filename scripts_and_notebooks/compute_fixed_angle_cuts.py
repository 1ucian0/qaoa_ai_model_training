# Copyright IBM Corp. 2026
# License: Apache-2.0

"""
Evaluate published fixed angles and write one
result file per instance under fixed_angle_cuts/,
in the format expected by training/eval_step.py.

Angles come from a JSON file keyed by degree and QAOA depth p, e.g.
https://github.com/danlkv/fixed-angle-QAOA/blob/master/angles_regular_graphs.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qaoa_training_pipeline.evaluation import PPEvaluator
from qaoa_training_pipeline.evaluation.efficient_depth_one import EfficientDepthOneEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator, load_graph

PPEVALUATOR_CONFIG = {"max_weight": 4, "min_abs_coeff": 1e-5}


def lookup_angles(angles_by_degree: dict, degree: int, depth: int, fallback: str) -> dict | None:
    """Angles for (degree, depth), or the nearest available degree if fallback='nearest'."""
    available = [int(d) for d in angles_by_degree if str(depth) in angles_by_degree[d]]
    if not available:
        return None

    if fallback == "nearest":
        degree = min(available, key=lambda d: abs(d - degree))
    elif degree not in available:
        return None

    return angles_by_degree[str(degree)][str(depth)]


def evaluate_cut(graph, betas: list[float], gammas: list[float], depth: int) -> float:
    """QAOA cut value at the given angles, matching QAOAEvalMixin.evaluate_qaoa."""
    cost_op = graph_to_operator(graph, pre_factor=-0.5)

    if depth == 1:
        energy = EfficientDepthOneEvaluator().evaluate(cost_op=cost_op, params=[betas[0], gammas[0]])
    else:
        energy = PPEvaluator(PPEVALUATOR_CONFIG).evaluate(cost_op=cost_op, params=[*betas, *gammas])

    sum_weights = sum(data.get("weight", 1.0) for _, _, data in graph.edges(data=True))
    return float(energy) + 0.5 * float(sum_weights)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--angles_file", type=Path, required=True,
                   help="JSON file with published angles, keyed by degree then depth p, "
                        "e.g. angles_regular_graphs.json from danlkv/fixed-angle-QAOA.")
    p.add_argument("--instance_dir", type=Path, required=True,
                   help="Directory of graph instance JSON files for one instance_set, "
                        "e.g. <graph_dir>/instances/random_regular/.")
    p.add_argument("--instance_set", required=True,
                   help="Name used for the output subdirectory, e.g. random_regular.")
    p.add_argument("--depth", type=int, required=True, help="QAOA depth p.")
    p.add_argument("--degree_fallback", choices=["exact", "nearest"], default="exact",
                   help="'exact' skips instances outside the angle table's degree range; "
                        "'nearest' uses the closest available degree instead. Default: exact.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Defaults to <graph_dir>/fixed_angle_cuts, inferred as the parent "
                        "of --instance_dir's grandparent (instances/<set>/../..).")
    args = p.parse_args()

    angles_by_degree = json.loads(args.angles_file.read_text())
    out_dir = args.out_dir or args.instance_dir.parents[1] / "fixed_angle_cuts"
    out_dir = out_dir / f"p_{args.depth}" / args.instance_set
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    for graph_path in sorted(args.instance_dir.glob("*.json")):
        graph = load_graph(graph_path)
        degree = int(round(np.mean([d for _, d in graph.degree])))

        angles = lookup_angles(angles_by_degree, degree, args.depth, args.degree_fallback)
        if angles is None:
            print(f"Skipping {graph_path.stem}: no angles for degree={degree}, p={args.depth}")
            skipped += 1
            continue

        betas, gammas = angles["beta"], angles["gamma"]
        cut_value = evaluate_cut(graph, betas, gammas, args.depth)

        out_file = out_dir / f"{graph_path.stem}.json"
        out_file.write_text(json.dumps(
            {"fixed_angle_cut": cut_value, "betas": betas, "gammas": gammas}, indent=2
        ))
        written += 1

    print(f"Wrote {written} fixed-angle result(s) to {out_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
