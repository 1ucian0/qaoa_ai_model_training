# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Tests for building best_param.json from raw qaoa_training_pipeline results."""

import json
import sys

from scripts_and_notebooks import build_best_param


def _write_result(path, instance_file, stages):
    result = {"args": {"input": str(instance_file), "pre_factor": -0.5}, "cost_operator": []}
    result.update({str(i): stage for i, stage in enumerate(stages)})
    path.write_text(json.dumps(result))


def test_keeps_highest_energy_per_instance_and_depth(tmp_path, monkeypatch):
    instance_file = tmp_path / "instances" / "random_regular" / "g0.json"
    instance_file.parent.mkdir(parents=True)
    instance_file.write_text("{}")
    results = tmp_path / "results_p1" / "random_regular"
    results.mkdir(parents=True)
    _write_result(results / "a.json", instance_file, [
        {"optimized_qaoa_angles": [0.1, 0.2], "energy": 1.0},
        {"optimized_qaoa_angles": [0.3, 0.4, 0.5, 0.6], "energy": "NA"},
    ])
    _write_result(results / "b.json", instance_file, [
        {"optimized_qaoa_angles": [0.4, 0.6], "energy": 2.0},
        {"optimized_qaoa_angles": [0.1, 0.2, 0.3, 0.4], "energy": 3.0},
    ])

    monkeypatch.setattr(sys, "argv", [
        "build_best_param", "--results", str(tmp_path / "results_p1"), "--graphs_repo", str(tmp_path),
    ])
    build_best_param.main()

    p1 = json.loads((tmp_path / "results_p1" / "best_param.json").read_text())
    p2 = json.loads((tmp_path / "results_p2" / "best_param.json").read_text())
    assert p1 == {"random_regular": {"g0": {"1": {
        "energy": 2.0, "qaoa_angles": [0.4, 0.6], "result_file_name": "b.json"}}}}
    assert p2["random_regular"]["g0"]["2"]["energy"] == 3.0
