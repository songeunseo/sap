import math

import pytest
import torch

from lib.dlm_gradient_sensitivity import (
    TimestepSensitivityAccumulator,
    make_masked_state,
    mask_probability,
    midpoint_timesteps,
    official_dlm_loss,
)


def test_midpoints_and_mask_probability_are_official_values():
    assert midpoint_timesteps() == pytest.approx(
        (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    )
    assert mask_probability(0.05) == pytest.approx(0.05095)


def test_official_loss_uses_masked_sum_over_full_sequence_and_p_mask():
    logits = torch.zeros(1, 4, 2)
    clean = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[True, False, True, False]])
    loss = official_dlm_loss(logits, clean, mask, p_mask=0.5)
    assert loss.item() == pytest.approx(math.log(2.0))


def test_fixed_mask_state_is_reproducible_and_nonempty():
    clean = torch.arange(256).unsqueeze(0)
    first = make_masked_state(clean, 0.05, mask_id=999, seed=7)
    second = make_masked_state(clean, 0.05, mask_id=999, seed=7)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[1].any()
    assert torch.equal(first[0][~first[1]], clean[~first[1]])
    assert torch.equal(first[0][first[1]], torch.full_like(first[0][first[1]], 999))


def test_mask_free_draw_retries_with_incremented_seed(monkeypatch):
    draws = iter((torch.ones(4), torch.tensor([0.0, 1.0, 1.0, 1.0])))

    def fake_rand(*args, **kwargs):
        return next(draws).reshape(args[0])

    monkeypatch.setattr(torch, "rand", fake_rand)
    clean = torch.arange(4).unsqueeze(0)
    masked, mask, _ = make_masked_state(clean, 0.5, mask_id=999, seed=7)
    assert mask.tolist() == [[True, False, False, False]]
    assert masked[0, 0].item() == 999


def test_official_loss_rejects_invalid_inputs():
    logits = torch.zeros(1, 2, 2)
    clean = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="masked token"):
        official_dlm_loss(logits, clean, torch.zeros_like(clean, dtype=torch.bool), 0.5)
    with pytest.raises(ValueError, match="p_mask"):
        official_dlm_loss(logits, clean, torch.ones_like(clean, dtype=torch.bool), 0.0)
    with pytest.raises(ValueError, match="shape"):
        official_dlm_loss(logits, clean, torch.ones(1, 3, dtype=torch.bool), 0.5)


def _add_scalar_states(accumulator, values, split):
    for value in values:
        accumulator.add_state({"w": torch.tensor(value)}, split)


def test_accumulator_updates_once_per_timestep_with_population_sigma():
    accumulator = TimestepSensitivityAccumulator(
        {"w": torch.tensor(2.0)}, split_size=2, expected_timesteps=2
    )
    for a, b in [((1, 3), (5, 7)), ((2, 4), (6, 8))]:
        _add_scalar_states(accumulator, a, "a")
        _add_scalar_states(accumulator, b, "b")
        accumulator.finish_timestep()

    result = accumulator.finalize()
    assert result["full"]["mu"]["w"].item() == 102
    assert result["full"]["sigma"]["w"].item() == 18
    assert result["a"]["sigma"]["w"].item() == 10
    assert result["b"]["sigma"]["w"].item() == 26
    assert result["update_count"] == 2


def test_accumulator_rejects_incomplete_or_invalid_timesteps():
    accumulator = TimestepSensitivityAccumulator(
        {"w": torch.tensor(2.0)}, split_size=4, expected_timesteps=1
    )
    with pytest.raises(ValueError, match="complete"):
        accumulator.finish_timestep()
    with pytest.raises(ValueError, match="split"):
        accumulator.add_state({"w": torch.tensor(1.0)}, "c")
    with pytest.raises(ValueError, match="keys"):
        accumulator.add_state({}, "a")
    with pytest.raises(ValueError, match="finite"):
        accumulator.add_state({"w": torch.tensor(float("nan"))}, "a")
    _add_scalar_states(accumulator, [1, 2, 3, 4], "a")
    with pytest.raises(ValueError, match="split_size"):
        accumulator.add_state({"w": torch.tensor(5.0)}, "a")
    with pytest.raises(ValueError, match="complete"):
        accumulator.finalize()
