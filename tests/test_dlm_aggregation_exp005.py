import json
from pathlib import Path

import pytest

from experiments.dlm_loss_aggregation.exp005.run import (
    load_config,
    summarize_symmetric_token_weights,
)


def test_exp005_config_is_abs_only_with_two_new_methods():
    config = load_config("experiments/dlm_loss_aggregation/exp005/config.json")

    assert config["experiment"] == "EXP-005"
    assert config["weighting"]["rho"] == 0.5
    assert config["scoring"]["aggregations"] == ["abs"]
    assert config["evaluation"]["methods"] == ["SYM-REVEAL-ABS", "SYM-REMAIN-ABS"]
    assert config["statistics"]["primary_comparison"] == [
        "SYM-REVEAL-ABS", "SYM-REMAIN-ABS"
    ]


def test_symmetric_summary_records_q_and_equal_intervention_magnitudes():
    summary = summarize_symmetric_token_weights(
        {"masked_count": 4, "reveal_count": 1, "remain_count": 3}, rho=0.5
    )

    assert summary["q"] == pytest.approx(0.25)
    assert summary["symmetric_reveal"]["reveal_alpha"] == pytest.approx(1.5)
    assert summary["symmetric_reveal"]["remain_alpha"] == pytest.approx(5 / 6)
    assert summary["symmetric_remain"]["reveal_alpha"] == pytest.approx(0.5)
    assert summary["symmetric_remain"]["remain_alpha"] == pytest.approx(7 / 6)
    assert summary["symmetric_reveal"]["normalized_mean"] == 1.0
    assert summary["symmetric_remain"]["normalized_mean"] == 1.0
    assert summary["symmetric_reveal"]["reveal_alpha"] + summary["symmetric_remain"]["reveal_alpha"] == 2.0


def test_exp005_reuses_exp004_partition_artifact_path():
    config = json.loads(Path("experiments/dlm_loss_aggregation/exp005/config.json").read_text())
    assert config["source_exp004"]["partition_summary"] == (
        "experiments/dlm_loss_aggregation/exp004/token_partition_summary.json"
    )
