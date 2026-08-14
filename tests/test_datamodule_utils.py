# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Unit tests for datamodule.utils angle rescaling helpers."""

import pytest

torch = pytest.importorskip("torch")

from datamodule.utils import angles_to_target, undo_gamma_rescale


def test_angles_to_target_rescales_only_gammas():
    p = 2
    angles = [0.1, 0.2, 0.3, 0.4]  # betas, gammas
    rescale_a = 2.0

    out = angles_to_target(angles, p, rescale_a)

    assert torch.allclose(out[:p], torch.tensor([0.1, 0.2]))
    assert torch.allclose(out[p:], torch.tensor([0.6, 0.8]))


def test_angles_to_target_and_undo_gamma_rescale_round_trip():
    p = 2
    angles = [0.1, 0.2, 0.3, 0.4]
    rescale_a = 2.0

    rescaled = angles_to_target(angles, p, rescale_a)
    restored = undo_gamma_rescale(rescaled, p, rescale_a)

    assert torch.allclose(restored, torch.tensor(angles, dtype=torch.float32))


def test_angles_to_target_length_mismatch_raises():
    with pytest.raises(ValueError, match="must have length 2\\*p"):
        angles_to_target([0.1, 0.2, 0.3], p=2, rescale_a=1.0)
