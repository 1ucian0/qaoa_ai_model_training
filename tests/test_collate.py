# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Unit tests for datamodule.collate.qaoa_collate_fn."""

import pytest

torch = pytest.importorskip("torch")

from datamodule.collate import qaoa_collate_fn


def test_empty_batch_returns_empty_dicts():
    assert qaoa_collate_fn([]) == ({}, {})


def test_variable_size_edges_kept_as_list():
    """Edges differ in count per graph, so they must stay an unstacked list."""
    batch = [
        ({"num_nodes": torch.tensor(4.0), "edges": torch.zeros(3, 2)}, {}),
        ({"num_nodes": torch.tensor(5.0), "edges": torch.zeros(5, 2)}, {}),
    ]

    x, _ = qaoa_collate_fn(batch)

    assert x["num_nodes"].shape == (2,)
    assert isinstance(x["edges"], list)
    assert [e.shape[0] for e in x["edges"]] == [3, 5]