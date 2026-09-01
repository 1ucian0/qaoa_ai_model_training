# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Generate a tiny synthetic graph_dir for smoke-testing the pipeline end to end.

This does not require real QAOA optimization results. It creates a handful of
small random graphs with made-up (not optimized) angles, purely so the
training and testing scripts can be exercised without a real dataset. Do not
use the resulting checkpoints for anything beyond verifying that the pipeline
runs.

Usage:
    python -m scripts_and_notebooks.make_smoke_test_data --out_dir /tmp/qaoa_smoke_test
"""

import argparse
import json
import random
from pathlib import Path

INSTANCE_SET = "smoke_test"
NUM_INSTANCES = 12
P = 1


def _make_graph(rng: random.Random, num_nodes: int) -> dict:
    edges = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < 0.5:
                edges.append({"nodes": [u, v], "weight": 1})
    if not edges:
        edges.append({"nodes": [0, 1], "weight": 1})
    return {"edge list": edges}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    graph_dir = args.out_dir.resolve()
    instance_dir = graph_dir / "instances" / INSTANCE_SET
    instance_dir.mkdir(parents=True, exist_ok=True)

    results_dir = graph_dir / f"results_p{P}"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_param = {INSTANCE_SET: {}}
    for i in range(NUM_INSTANCES):
        instance_id = f"instance_{i:03d}"
        num_nodes = rng.randint(6, 10)
        graph = _make_graph(rng, num_nodes)
        (instance_dir / f"{instance_id}.json").write_text(json.dumps(graph, indent=2))

        best_param[INSTANCE_SET][instance_id] = {
            str(P): {
                "energy": rng.uniform(-5.0, -1.0),
                "qaoa_angles": [rng.uniform(0, 1.5) for _ in range(2 * P)],
                "result_file_name": f"{instance_id}.json",
            }
        }

    (results_dir / "best_param.json").write_text(json.dumps(best_param, indent=2))

    print(f"Wrote {NUM_INSTANCES} synthetic instances to {graph_dir}")
    print("Next steps (run from the repository root):")
    print(
        "  python -m scripts_and_notebooks.testset_generation "
        f"--graphs_repo {graph_dir} --instance_sets {INSTANCE_SET}"
    )
    print(
        "  python -m scripts_and_notebooks.build_per_p_split "
        f"--graphs_repo {graph_dir} --target_p 1"
    )


if __name__ == "__main__":
    main()
