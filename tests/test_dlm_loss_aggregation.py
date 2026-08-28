import torch
import pytest

from experiments.dlm_loss_aggregation.core import (
    EffectAccumulator,
    mask_sha256,
    pack_mask,
    pairwise_diagnostics,
    rowwise_mask,
    unpack_mask,
)


def test_effect_accumulator_implements_signed_absolute_and_square_means():
    accumulator = EffectAccumulator((2, 2))
    accumulator.add(torch.tensor([[1.0, -2.0], [-3.0, 4.0]]))
    accumulator.add(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]))

    scores = accumulator.finalize()

    assert torch.equal(scores["sum"], torch.zeros(2, 2))
    assert torch.equal(scores["abs"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(scores["square"], torch.tensor([[1.0, 4.0], [9.0, 16.0]]))
    assert accumulator.count == 2


def test_effect_accumulator_accepts_future_scalar_state_weights():
    accumulator = EffectAccumulator((1, 2))
    accumulator.add(torch.tensor([[1.0, -2.0]]), state_weight=1.0)
    accumulator.add(torch.tensor([[4.0, -8.0]]), state_weight=3.0)

    scores = accumulator.finalize()

    assert torch.equal(scores["sum"], torch.tensor([[3.25, -6.5]]))
    assert torch.equal(scores["abs"], torch.tensor([[3.25, 6.5]]))
    assert torch.equal(scores["square"], torch.tensor([[12.25, 49.0]]))


def test_rowwise_mask_prunes_signed_scores_in_stable_ascending_order():
    score = torch.tensor([[-4.0, 3.0, -2.0, 1.0], [2.0, 2.0, 1.0, 3.0]])

    mask = rowwise_mask(score, 0.5)

    assert torch.equal(
        mask,
        torch.tensor([[True, False, True, False], [True, False, True, False]]),
    )
    assert torch.equal(mask.sum(dim=1), torch.tensor([2, 2]))


def test_single_state_absolute_and_square_masks_may_be_identical():
    effect = torch.tensor([[-3.0, 1.0, -2.0, 4.0]])

    absolute = rowwise_mask(effect.abs(), 0.5)
    square = rowwise_mask(effect.square(), 0.5)

    assert torch.equal(absolute, square)


def test_compact_mask_round_trip_and_hash_are_stable():
    mask = torch.tensor([[True, False, True], [False, True, False]])

    payload = pack_mask(mask)

    assert unpack_mask(payload).equal(mask)
    assert mask_sha256(payload) == mask_sha256(pack_mask(mask.clone()))
    assert len(payload["bits"]) == 1


def test_pairwise_diagnostics_use_signed_ranks_and_exact_full_mask_xor():
    diagnostics = pairwise_diagnostics(
        {
            "sum": torch.tensor([[-4.0, 1.0, -2.0, 3.0]]),
            "abs": torch.tensor([[4.0, 1.0, 2.0, 3.0]]),
            "square": torch.tensor([[16.0, 1.0, 4.0, 9.0]]),
        }
    )

    pairs = {(row["left"], row["right"]): row for row in diagnostics["pairs"]}
    assert pairs[("sum", "abs")]["spearman"] == pytest.approx(-0.4)
    assert pairs[("sum", "square")]["spearman"] == pytest.approx(-0.4)
    assert pairs[("abs", "square")]["spearman"] == pytest.approx(1.0)
    assert pairs[("sum", "abs")]["mask_xor"] == pytest.approx(0.5)
    assert pairs[("abs", "square")]["mask_xor"] == 0.0
    assert diagnostics["negative_sum_fraction"] == 0.5
    assert diagnostics["sign_consistency"]["mean"] == pytest.approx(1.0)
    assert diagnostics["spike_ratio"]["mean"] == pytest.approx(1.0)


def test_sampled_spearman_reuses_deterministic_indices_but_xor_stays_exact():
    scores = {
        "sum": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "abs": torch.arange(19, -1, -1, dtype=torch.float32).reshape(2, 10),
        "square": torch.arange(20, dtype=torch.float32).square().reshape(2, 10),
    }

    first = pairwise_diagnostics(scores, spearman_sample_size=5, seed=7)
    second = pairwise_diagnostics(scores, spearman_sample_size=5, seed=7)

    assert first == second
    assert {row["sample_size"] for row in first["pairs"]} == {5}
    assert len({row["sample_indices_sha256"] for row in first["pairs"]}) == 1
    assert first["pairs"][0]["mask_xor"] == 1.0


def test_constant_score_correlation_is_recorded_as_undefined():
    diagnostics = pairwise_diagnostics(
        {
            "sum": torch.zeros(1, 4),
            "abs": torch.ones(1, 4),
            "square": torch.ones(1, 4),
        }
    )

    assert diagnostics["pairs"][0]["spearman"] is None
    assert diagnostics["pairs"][0]["reason"] == "constant score"
