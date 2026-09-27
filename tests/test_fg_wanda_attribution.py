import pytest
import torch

from experiments.fg_wanda_prototype1.attribution_core import (
    attribution_decomposition,
    build_attribution_masks,
    paired_correctness,
    three_way_correctness,
)


def test_build_attribution_masks_changes_only_frozen_target():
    standard = {
        "block_00.q_proj": torch.tensor([[True, False, True, False]]),
        "block_31.ff_out": torch.tensor([[True, True, False, False]]),
    }
    dlmw_target = torch.tensor([[False, True, True, False]])
    fg_target = torch.tensor([[False, False, True, True]])

    ours_dlmw, changed_vs_wanda, changed_vs_fg = build_attribution_masks(
        standard,
        dlmw_target,
        fg_target,
    )

    assert changed_vs_wanda == ["block_31.ff_out"]
    assert changed_vs_fg == ["block_31.ff_out"]
    assert torch.equal(ours_dlmw["block_00.q_proj"], standard["block_00.q_proj"])
    assert torch.equal(ours_dlmw["block_31.ff_out"], dlmw_target)


def test_build_attribution_masks_rejects_wrong_target_shape_or_density():
    standard = {"block_31.ff_out": torch.tensor([[True, True, False, False]])}
    fg_target = torch.tensor([[False, False, True, True]])

    with pytest.raises(ValueError, match="shape"):
        build_attribution_masks(standard, torch.tensor([[True, False]]), fg_target)
    with pytest.raises(ValueError, match="pruned count"):
        build_attribution_masks(
            standard,
            torch.tensor([[True, True, True, False]]),
            fg_target,
        )


def test_three_way_correctness_returns_complete_eight_cell_table():
    wanda = [True, True, True, True, False, False, False, False]
    dlmw = [True, True, False, False, True, True, False, False]
    fg = [True, False, True, False, True, False, True, False]

    table = three_way_correctness(wanda, dlmw, fg)

    assert table == {
        "W0_D0_F0": 1,
        "W0_D0_F1": 1,
        "W0_D1_F0": 1,
        "W0_D1_F1": 1,
        "W1_D0_F0": 1,
        "W1_D0_F1": 1,
        "W1_D1_F0": 1,
        "W1_D1_F1": 1,
    }


def test_paired_correctness_and_decomposition_preserve_direction():
    left = [True, True, False, False]
    right = [True, False, True, True]

    paired = paired_correctness(left, right, left_name="Wanda", right_name="DLMW")
    assert paired["both_correct"] == 1
    assert paired["Wanda_only"] == 1
    assert paired["DLMW_only"] == 2
    assert paired["correct_delta_DLMW_minus_Wanda"] == 1
    assert paired["delta_percentage_points_DLMW_minus_Wanda"] == 25.0
    assert paired["mcnemar_exact_pvalue"] == 1.0

    decomp = attribution_decomposition(wanda_correct=10, dlmw_correct=13, fg_correct=15, n=100)
    assert decomp["total_FG_minus_Wanda"] == {"examples": 5, "percentage_points": 5.0}
    assert decomp["calibration_DLMW_minus_Wanda"] == {"examples": 3, "percentage_points": 3.0}
    assert decomp["fisher_incremental_FG_minus_DLMW"] == {"examples": 2, "percentage_points": 2.0}
