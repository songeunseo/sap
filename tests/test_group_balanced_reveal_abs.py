import math

import pytest
import torch

from experiments.group_balanced_reveal_abs.run import (
    exact_paired_binomial_pvalue,
    group_balanced_token_weights,
    summarize_group_balance,
)


def test_group_balanced_weights_split_mass_equally():
    mask = torch.tensor([[True, True, True, True, True, False]])
    reveal = torch.tensor([[True, False, False, False, False, False]])

    alpha = group_balanced_token_weights(mask, reveal)

    assert alpha[reveal].sum().item() == pytest.approx(2.5)
    assert alpha[mask & ~reveal].sum().item() == pytest.approx(2.5)
    assert alpha[mask].mean().item() == pytest.approx(1.0)
    assert alpha[reveal].item() / alpha[mask & ~reveal][0].item() == pytest.approx(4.0)


def test_group_balanced_weighted_sum_matches_direct_group_mean():
    mask = torch.tensor([[True, True, True, True, True]])
    reveal = torch.tensor([[True, False, False, False, False]])
    losses = torch.tensor([2.0, 1.0, 3.0, 5.0, 7.0])
    alpha = group_balanced_token_weights(mask, reveal)

    weighted = (losses * alpha[mask]).sum()
    direct = mask.sum() * 0.5 * (losses[:1].mean() + losses[1:].mean())

    assert weighted.item() == pytest.approx(direct.item())


def test_all_state_summary_exposes_required_fields():
    row = summarize_group_balance(
        {"state_index": 7, "masked_count": 133, "reveal_count": 1, "remain_count": 132}
    )
    assert row["masked_count"] == 133
    assert row["reveal_count"] == 1
    assert row["remain_count"] == 132
    assert row["reveal_group_share"] == pytest.approx(0.5)
    assert row["remain_group_share"] == pytest.approx(0.5)
    assert row["alpha_ratio"] == pytest.approx(132.0)


@pytest.mark.parametrize(
    ("left_only", "right_only", "expected"),
    [(0, 0, 1.0), (1, 0, 1.0), (2, 0, 0.5), (3, 0, 0.25), (2, 2, 1.0)],
)
def test_exact_paired_binomial_uses_only_discordant_examples(left_only, right_only, expected):
    assert exact_paired_binomial_pvalue(left_only, right_only) == pytest.approx(expected)


def test_group_balanced_weights_reject_empty_group():
    mask = torch.tensor([[True, True]])
    reveal = torch.tensor([[False, False]])
    with pytest.raises(ValueError, match="both reveal and remain"):
        group_balanced_token_weights(mask, reveal)
