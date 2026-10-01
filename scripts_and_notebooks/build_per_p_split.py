# Copyright IBM Corp. 2026
# License: Apache-2.0

"""
Snapshot the master split filtered to instances available at certain p.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _available_instance_ids(best_param_file: Path) -> set[str]:
    """Return instance ids available in the best_param summary file."""
    data = json.loads(best_param_file.read_text())
    out: set[str] = set()
    for v in data.values():
        if isinstance(v, dict):
            for k in v.keys():
                out.add(Path(k).stem)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs_repo", type=Path, required=True)
    p.add_argument("--target_p", type=int, required=True)
    p.add_argument("--splits_dir", type=Path, default=None,
                   help="Defaults to <graphs_repo>/splits/. Per-p outputs land in <splits_dir>/p_{p}/.")
    args = p.parse_args()

    repo = args.graphs_repo.resolve()
    splits_dir = (args.splits_dir or repo / "splits").resolve()
    results_dir = splits_dir / f"p_{args.target_p}"
    results_dir.mkdir(parents=True, exist_ok=True)

    repo_results_dir = repo / f"results_p{args.target_p}"
    best_param_file = repo_results_dir / "best_param.json"
    master_test = splits_dir / "test_split.json"
    master_split = splits_dir / "train_val_split.json"
    for f in (best_param_file, master_test, master_split):
        if not f.exists():
            sys.exit(f"Error: required file missing: {f}\n"
                     f"  Build results_p{args.target_p}/best_param.json (see README.md), "
                     f"then run scripts_and_notebooks/testset_generation.py to produce the master splits.")

    available = _available_instance_ids(best_param_file)

    test_data = json.loads(master_test.read_text())
    split_data = json.loads(master_split.read_text())

    # Filter master indices down to what's available at this p.
    test_ids   = [iid for iid in test_data["test_indices"]  if iid in available]
    train_ids  = [iid for iid in split_data["train_indices"] if iid in available]
    val_ids    = [iid for iid in split_data["val_indices"]   if iid in available]

    test_data["test_indices"] = test_ids
    test_data.setdefault("meta", {})["target_p"] = int(args.target_p)
    split_data["train_indices"] = train_ids
    split_data["val_indices"] = val_ids
    split_data.setdefault("meta", {})["target_p"] = int(args.target_p)
    # norm_stats already in split_data from the master file — no recomputation needed.

    out_test = results_dir / "test_split.json"
    out_train_val = results_dir / "train_val_split.json"
    out_test.write_text(json.dumps(test_data, indent=2))
    out_train_val.write_text(json.dumps(split_data, indent=2))

    print(f"Snapshot for p={args.target_p}: test={len(test_ids)}, train={len(train_ids)}, val={len(val_ids)}")
    print(f"  ({len(available)} instances available in best_param)")
    print(f"Saved: {out_train_val}")
    print(f"       {out_test}")


if __name__ == "__main__":
    main()