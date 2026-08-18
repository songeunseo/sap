import math

import pytest
import torch

from lib.dlm_gradient_sensitivity import (
    TimestepSensitivityAccumulator,
    make_masked_state,
    mask_probability,
    midpoint_timesteps,
    official_dlm_loss,
    jaccard,
    load_mask_block,
    pack_mask,
    rowwise_prune_mask,
    sampled_spearman,
    save_mask_block,
    unpack_mask,
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


def test_one_token_mask_retries_until_nonempty():
    masked, mask, _ = make_masked_state(torch.tensor([[42]]), 0.05, mask_id=999, seed=0)
    assert mask.tolist() == [[True]]
    assert masked.item() == 999


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


# Mutation caught: choosing a global threshold or rounding the prune count.
@pytest.mark.parametrize(
    ("score", "sparsity", "per_row"),
    [
        (torch.arange(15, dtype=torch.float32).reshape(3, 5), 0.0, 0),
        (torch.arange(15, dtype=torch.float32).reshape(3, 5), 0.5, 2),
        (torch.arange(21, dtype=torch.float32).reshape(3, 7), 0.6, 4),
        (torch.arange(44, dtype=torch.float32).reshape(4, 11), 0.75, 8),
    ],
)
def test_rowwise_prune_mask_prunes_floor_count_in_each_row(score, sparsity, per_row):
    mask = rowwise_prune_mask(score, sparsity)
    assert mask.dtype is torch.bool
    assert mask.shape == score.shape
    assert mask.sum(dim=1).tolist() == [per_row] * score.shape[0]


# Mutation caught: dropping stable sorting makes equal scores select arbitrary columns.
def test_rowwise_prune_mask_breaks_ties_by_column_index():
    mask = rowwise_prune_mask(torch.ones(2, 5), 0.4)
    assert mask.tolist() == [[True, True, False, False, False]] * 2


# Mutation caught: truncating non-byte-aligned tail bits corrupts stored masks.
def test_pack_mask_round_trips_non_byte_aligned_mask():
    mask = torch.tensor(
        [[True, False, True, False, False, True, False, True, False, False, True],
         [False, True, False, True, True, False, True, False, True, True, False],
         [True, True, False, False, True, False, False, True, True, False, False]]
    )
    packed = pack_mask(mask)
    assert packed["byte_length"] == 5
    assert torch.equal(unpack_mask(packed), mask)


# Mutation caught: accepting checksum or geometry changes loads an artifact for another mask.
def test_unpack_mask_rejects_corrupt_bits_or_shape():
    packed = pack_mask(torch.tensor([[True, False, True], [False, True, False]]))
    corrupt_bits = dict(packed, bits=bytes([packed["bits"][0] ^ 1]))
    corrupt_shape = dict(packed, shape=[3, 2])
    with pytest.raises(ValueError, match="checksum"):
        unpack_mask(corrupt_bits)
    with pytest.raises(ValueError, match="checksum"):
        unpack_mask(corrupt_shape)


# Mutation caught: replacing the destination before serialization loses the last valid artifact.
def test_save_mask_block_keeps_existing_artifact_when_write_fails(tmp_path):
    path = tmp_path / "masks.json"
    original = {"module": torch.tensor([[True, False, True]])}
    save_mask_block(path, original, {"run": "old"})
    with pytest.raises(TypeError):
        save_mask_block(path, {"module": torch.tensor([[False, True, False]])}, {"bad": {1}})
    loaded, metadata = load_mask_block(path)
    assert torch.equal(loaded["module"], original["module"])
    assert metadata == {"run": "old"}


# Mutation caught: ordinal ranks or nondeterministic sampling change tied-rank correlations.
def test_sampled_spearman_uses_average_tie_ranks_and_deterministic_indices():
    identical = sampled_spearman(torch.arange(10), torch.arange(10), 4, seed=17)
    reversed_ranks = sampled_spearman(torch.arange(4), torch.arange(3, -1, -1), 4, seed=17)
    tied = sampled_spearman(
        torch.tensor([1.0, 1.0, 2.0, 3.0]), torch.tensor([1.0, 1.0, 3.0, 2.0]), 4, seed=17
    )
    assert identical == pytest.approx(1.0)
    assert identical.sample_indices.tolist() == [9, 7, 0, 5]
    assert reversed_ranks == pytest.approx(-1.0)
    assert tied == pytest.approx(7 / 9)


# Mutation caught: converting undefined constant-vector correlation to a plausible zero hides failure.
def test_sampled_spearman_reports_constant_vectors_as_nan():
    rho = sampled_spearman(torch.ones(4), torch.arange(4), 4, seed=0)
    assert math.isnan(rho)
    assert rho.reason == "constant vector"


# Mutation caught: dividing by the left mask count instead of set union misstates overlap diagnostics.
def test_jaccard_captures_mask_and_row_change_diagnostics():
    left = torch.tensor([[True, True, False, False], [True, False, True, False]])
    right = torch.tensor([[True, False, True, False], [True, False, True, False]])
    assert jaccard(left, right) == pytest.approx(0.6)
    assert left.ne(right).any(dim=1).float().mean().item() == pytest.approx(0.5)

    sigma_a = torch.arange(100)
    sigma_b = torch.arange(99, -1, -1)
    top_a = sigma_a >= 99
    top_b = sigma_b >= 99
    assert jaccard(top_a, top_b) == 0.0
