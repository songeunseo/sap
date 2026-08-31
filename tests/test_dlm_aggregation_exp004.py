import math
import json
from pathlib import Path

import pytest
import torch

from experiments.dlm_loss_aggregation.exp004.run import (
    MagnitudeAccumulator,
    _report,
    compare_score_masks,
    holm_adjust,
    independent_weight_effects,
    load_config,
    paired_comparison,
    freeze_token_partition,
    require_disk_capacity,
    summarize_mask_diagnostics,
    summarize_token_weights,
    validate_uniform_reuse,
    token_weights,
    validate_uniform_reproduction,
    validate_resumable_evaluation,
    validate_partition_artifact,
    weighted_dlm_loss,
    weighted_dlm_losses,
    write_packed_mask,
)

from experiments.dlm_loss_aggregation.core import unpack_mask


def test_freeze_partition_uses_remaining_schedule_and_partitions_every_mask():
    logits = torch.full((1, 6, 4), -10.0)
    logits[0, 0, 1] = math.log(9)
    logits[0, 1, 2] = math.log(8)
    logits[0, 2, 3] = math.log(7)
    logits[0, 3, 1] = math.log(6)
    noisy = torch.tensor([[99, 99, 99, 99, 4, 5]])

    partition = freeze_token_partition(
        logits, noisy, mask_id=99, p_mask=0.25, denoising_steps=8
    )

    assert partition["steps_remaining"] == 2
    assert partition["masked_count"] == 4
    assert partition["reveal_count"] == 2
    assert partition["remain_count"] == 2
    assert partition["masked_indices"] == [0, 1, 2, 3]
    assert partition["reveal_indices"] == [0, 1]
    assert partition["remain_indices"] == [2, 3]
    assert set(partition["reveal_indices"]).isdisjoint(partition["remain_indices"])
    assert sorted(partition["reveal_indices"] + partition["remain_indices"]) == partition["masked_indices"]
    assert [row["position"] for row in partition["masked_predictions"]] == [0, 1, 2, 3]
    assert [row["token_id"] for row in partition["masked_predictions"]] == [1, 2, 3, 1]
    assert all(0 < row["confidence"] <= 1 for row in partition["masked_predictions"])


def test_partition_confidence_matches_generation_softmax_in_logits_dtype():
    logits = torch.tensor([[[1.1015625, 0.0]]], dtype=torch.bfloat16)
    noisy = torch.tensor([[99]])

    partition = freeze_token_partition(logits, noisy, 99, p_mask=1.0)
    predicted = logits.argmax(dim=-1)
    expected = torch.nn.functional.softmax(logits, dim=-1).gather(
        -1, predicted.unsqueeze(-1)
    ).squeeze().item()

    assert partition["masked_predictions"][0]["confidence"] == expected


def test_partition_topk_matches_full_sequence_generator_tie_breaking():
    logits = torch.zeros((1, 5, 2))
    noisy = torch.tensor([[99, 1, 99, 1, 99]])

    partition = freeze_token_partition(logits, noisy, 99, p_mask=1.0)
    confidence = torch.full((1, 5), 0.5)
    generator_confidence = torch.where(noisy.eq(99), confidence, -torch.inf)
    expected = torch.topk(generator_confidence[0], k=1).indices.tolist()

    assert partition["reveal_indices"] == expected


def test_token_weights_normalize_each_condition_to_masked_mean_one():
    mask = torch.tensor([[True, True, True, True, False]])
    reveal = torch.tensor([[True, True, False, False, False]])

    uniform = token_weights(mask, reveal, "uniform")
    reveal_up = token_weights(mask, reveal, "reveal")
    remain_up = token_weights(mask, reveal, "remain")

    assert torch.equal(uniform[mask], torch.ones(4))
    assert torch.equal(reveal_up[mask], torch.tensor([4 / 3, 4 / 3, 2 / 3, 2 / 3]))
    assert torch.equal(remain_up[mask], torch.tensor([2 / 3, 2 / 3, 4 / 3, 4 / 3]))
    assert reveal_up[mask].mean().item() == 1.0
    assert remain_up[mask].mean().item() == 1.0
    assert not uniform[~mask].any().item()


def test_weighted_loss_changes_only_token_terms_not_dlm_normalization():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    clean = torch.tensor([[0, 0, 1]])
    mask = torch.tensor([[True, True, False]])
    alpha = torch.tensor([[1.5, 0.5, 0.0]])

    loss = weighted_dlm_loss(logits, clean, mask, p_mask=0.5, alpha=alpha)
    token_ce = torch.nn.functional.cross_entropy(
        logits[mask].float(), clean[mask], reduction="none"
    )

    assert loss.item() == pytest.approx((1.5 * token_ce[0] + 0.5 * token_ce[1]).item() / 0.5 / 3)


def test_three_weighted_losses_share_one_per_token_ce_vector_semantics():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    clean = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, True]])
    alphas = {
        "uniform": torch.tensor([[1.0, 1.0]]),
        "reveal": torch.tensor([[4 / 3, 2 / 3]]),
        "remain": torch.tensor([[2 / 3, 4 / 3]]),
    }

    losses = weighted_dlm_losses(logits, clean, mask, p_mask=0.5, alphas=alphas)
    token_ce = torch.nn.functional.cross_entropy(
        logits[mask].float(), clean[mask], reduction="none"
    )

    assert losses["uniform"] == (token_ce.sum() / 0.5 / 2)
    assert losses["reveal"] == ((token_ce * torch.tensor([4 / 3, 2 / 3])).sum() / 0.5 / 2)
    assert losses["remain"] == ((token_ce * torch.tensor([2 / 3, 4 / 3])).sum() / 0.5 / 2)


def test_condition_effects_share_graph_without_accumulating_parameter_gradients():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
    output = layer(torch.tensor([[4.0, 5.0]])).sum()

    effects = independent_weight_effects(
        {
            "uniform": output,
            "reveal": 2 * output,
            "remain": -output,
        },
        {"linear": layer.weight},
    )

    assert torch.equal(effects["uniform"]["linear"], torch.tensor([[-8.0, 15.0]]))
    assert torch.equal(effects["reveal"]["linear"], torch.tensor([[-16.0, 30.0]]))
    assert torch.equal(effects["remain"]["linear"], torch.tensor([[8.0, -15.0]]))
    assert layer.weight.grad is None


def test_condition_effect_multiplies_weight_and_gradient_in_cpu_float32():
    layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.fill_(1.1015625)
    value = torch.tensor([[1.1015625]], dtype=torch.bfloat16)
    output = layer(value).sum()

    effects = independent_weight_effects(
        {"uniform": output, "reveal": output, "remain": output},
        {"linear": layer.weight},
    )

    expected = -(layer.weight.detach().cpu().float() * value.cpu().float())
    assert torch.equal(effects["uniform"]["linear"], expected)


def test_condition_effects_reuse_supplied_frozen_cpu_weight_snapshot():
    layer = torch.nn.Linear(1, 1, bias=False)
    output = layer(torch.tensor([[3.0]])).sum()
    frozen = {"linear": torch.tensor([[5.0]])}

    effects = independent_weight_effects(
        {"uniform": output, "reveal": output, "remain": output},
        {"linear": layer.weight},
        frozen,
    )

    assert torch.equal(effects["uniform"]["linear"], torch.tensor([[-15.0]]))


def test_uniform_reuse_requires_identical_protocol_and_paired_examples():
    uniform = {
        "UNIFORM-ABS": [
            {
                "example_id": 0,
                "doc_hash": "d0",
                "prompt_hash": "p0",
                "target_hash": "t0",
                "evaluation_config_hash": "cfg",
            }
        ],
        "UNIFORM-SQUARE": [
            {
                "example_id": 0,
                "doc_hash": "d0",
                "prompt_hash": "p0",
                "target_hash": "t0",
                "evaluation_config_hash": "cfg",
            }
        ],
    }

    assert validate_uniform_reuse(uniform, "cfg", "cfg", expected_count=1)["passed"]

    with pytest.raises(ValueError, match="config hash"):
        validate_uniform_reuse(uniform, "cfg", "different", expected_count=1)
    uniform["UNIFORM-SQUARE"][0]["prompt_hash"] = "different"
    with pytest.raises(ValueError, match="paired examples"):
        validate_uniform_reuse(uniform, "cfg", "cfg", expected_count=1)


def test_disk_preflight_fails_before_writing_when_final_masks_do_not_fit(tmp_path):
    result = require_disk_capacity(
        tmp_path, mask_bytes_per_method=100, method_count=4, safety_bytes=50,
        free_bytes=450,
    )
    assert result["required_bytes"] == 450

    with pytest.raises(RuntimeError, match="insufficient disk"):
        require_disk_capacity(
            tmp_path, mask_bytes_per_method=100, method_count=4,
            safety_bytes=51, free_bytes=450,
        )


def test_magnitude_accumulator_keeps_normalized_scores_and_raw_scale_diagnostics():
    accumulator = MagnitudeAccumulator((1, 2))
    accumulator.add(torch.tensor([[1.0, -2.0]]), raw_scale=1.5)
    accumulator.add(torch.tensor([[-3.0, 4.0]]), raw_scale=2.0)

    scores, diagnostics = accumulator.finalize()

    assert torch.equal(scores["abs"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(scores["square"], torch.tensor([[5.0, 10.0]]))
    assert diagnostics["normalized_abs_mean"] == pytest.approx(2.5)
    assert diagnostics["raw_abs_mean"] == pytest.approx(4.625)
    assert diagnostics["normalized_square_mean"] == pytest.approx(7.5)
    assert diagnostics["raw_square_mean"] == pytest.approx(27.8125)


def test_uniform_reproduction_reports_small_mismatch_and_aborts_above_threshold():
    rows = [
        {"method": "abs", "num_weights": 1_000_000, "different": 1},
        {"method": "square", "num_weights": 1_000_000, "different": 0},
    ]

    result = validate_uniform_reproduction(
        rows, global_xor_threshold=1e-6, module_xor_threshold=1e-5
    )
    assert result["exact"] is False
    assert result["by_method"]["abs"]["xor"] == 1e-6

    rows[0]["different"] = 11
    with pytest.raises(RuntimeError, match="UNIFORM mask reproduction"):
        validate_uniform_reproduction(
            rows, global_xor_threshold=1e-6, module_xor_threshold=1e-5
        )


def test_score_mask_comparison_reports_exact_overlap_diagnostics():
    left_score = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right_score = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    left_mask = torch.tensor([[True, True, False, False]])
    right_mask = torch.tensor([[True, False, True, False]])

    result = compare_score_masks(left_score, right_score, left_mask, right_mask)

    assert result["spearman"] == pytest.approx(0.8)
    assert result["mask_xor"] == 0.5
    assert result["mask_iou"] == pytest.approx(1 / 3)
    assert result["topk_overlap"] == 0.5


def test_paired_comparison_uses_exact_two_sided_binomial_and_holm_adjustment():
    left = [True, True, True, False, True, False]
    right = [True, False, False, True, False, False]
    records = {
        "A": [{"correct": value} for value in left],
        "B": [{"correct": value} for value in right],
    }

    row = paired_comparison("A", "B", records)

    assert row["both_correct"] == 1
    assert row["a_only_correct"] == 3
    assert row["b_only_correct"] == 1
    assert row["both_wrong"] == 1
    assert row["accuracy_difference_pp"] == pytest.approx(100 * (4 / 6 - 2 / 6))
    assert row["p_exact_two_sided"] == 0.625

    adjusted = holm_adjust([{"p": 0.01}, {"p": 0.04}, {"p": 0.03}], "p")
    assert [item["p_holm"] for item in adjusted] == pytest.approx([0.03, 0.06, 0.06])


def test_exp004_config_rejects_a_changed_weight_ratio(tmp_path):
    path = Path("experiments/dlm_loss_aggregation/exp004/config.json")
    config = load_config(path)
    assert config["experiment"] == "EXP-004"

    config["weighting"]["raw_ratio"] = 3.0
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_ratio"):
        load_config(changed)

    config = load_config(path)
    config["statistics"]["paired_comparisons"].pop()
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="paired_comparisons"):
        load_config(changed)


def test_packed_mask_writer_checksums_and_reads_back_one_matrix(tmp_path):
    mask = torch.tensor([[True, False, True, False]])

    entry = write_packed_mask(tmp_path, "reveal_abs", 3, "attn.out", mask)

    payload = {"shape": entry["shape"], "bits": Path(entry["runtime_path"]).read_bytes()}
    assert unpack_mask(payload).equal(mask)
    assert entry["byte_length"] == 1
    assert len(entry["sha256"]) == 64


def test_resume_validation_rejects_predictions_from_a_stale_mask():
    row = {
        "status": "passed", "method": "REVEAL-ABS", "mask_hash": "old",
        "evaluation_config_hash": "cfg", "correct": 1, "num_examples": 2,
        "accuracy": 0.5, "rowwise_exact": True,
    }
    records = [
        {"correct": True, "evaluation_config_hash": "cfg"},
        {"correct": False, "evaluation_config_hash": "cfg"},
    ]
    mask_document = {"overall_sha256": "new"}

    with pytest.raises(ValueError, match="mask hash"):
        validate_resumable_evaluation(
            row, records, "REVEAL-ABS", mask_document, "cfg", 2
        )

    mask_document["overall_sha256"] = "old"
    assert validate_resumable_evaluation(
        row, records, "REVEAL-ABS", mask_document, "cfg", 2
    )


def test_partition_artifact_is_bound_to_each_exp001_state_identity():
    manifest = {
        "historical_state_sha256": "states",
        "mask_id": 99,
        "states": [{
            "timestep_index": 0, "timestep": 0.5, "sequence_index": 7,
            "mask_seed": 3, "p_mask": 0.5,
            "mask": [[True, False, True]],
        }],
    }
    state = {
        "state_index": 0, "timestep_index": 0, "timestep": 0.5,
        "sequence_index": 7, "mask_seed": 3, "steps_remaining": 128,
        "masked_count": 2, "reveal_count": 1, "remain_count": 1,
        "masked_indices": [0, 2], "reveal_indices": [0], "remain_indices": [2],
        "masked_predictions": [
            {"position": 0, "token_id": 1, "confidence": 0.9},
            {"position": 2, "token_id": 1, "confidence": 0.8},
        ],
    }
    core = {
        "version": 1, "source_state_sha256": "states",
        "dense_fingerprint": "d" * 64, "partition_rule": {}, "states": [state],
    }
    artifact = {
        **core,
        "sha256": __import__("hashlib").sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "summary": {"state_count": 1},
    }

    assert validate_partition_artifact(artifact, manifest, expected_count=1)["passed"]

    artifact["states"][0]["sequence_index"] = 8
    changed_core = {key: artifact[key] for key in core}
    artifact["sha256"] = __import__("hashlib").sha256(
        json.dumps(changed_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="state identity"):
        validate_partition_artifact(artifact, manifest, expected_count=1)


def test_token_weight_summary_records_raw_and_normalized_values():
    summary = summarize_token_weights(
        {"masked_count": 4, "reveal_count": 2, "remain_count": 2}
    )

    assert summary["uniform"]["normalized_mean"] == 1.0
    assert summary["reveal"]["raw_mean"] == 1.5
    assert summary["reveal"]["reveal_alpha"] == pytest.approx(4 / 3)
    assert summary["reveal"]["remain_alpha"] == pytest.approx(2 / 3)
    assert summary["remain"]["reveal_alpha"] == pytest.approx(2 / 3)
    assert summary["remain"]["remain_alpha"] == pytest.approx(4 / 3)


def test_mask_diagnostics_add_element_weighted_layer_type_and_global_rows():
    rows = [
        {
            "scope": "matrix", "pair": "A/B", "layer": 0, "module": "x.q_proj",
            "module_type": "q_proj", "num_weights": 2, "different": 1,
            "spearman": 1.0, "mask_iou": 0.5, "topk_overlap": 0.5,
        },
        {
            "scope": "matrix", "pair": "A/B", "layer": 0, "module": "x.v_proj",
            "module_type": "v_proj", "num_weights": 6, "different": 0,
            "spearman": 0.5, "mask_iou": 1.0, "topk_overlap": 1.0,
        },
    ]

    result = summarize_mask_diagnostics(rows)
    overall = next(row for row in result if row["scope"] == "global")

    assert overall["num_weights"] == 8
    assert overall["mask_xor"] == 0.125
    assert overall["spearman"] == pytest.approx(0.625)
    assert {row["scope"] for row in result} == {"matrix", "layer", "module_type", "global"}


def test_final_report_renders_all_required_result_and_statistics_sections():
    config = load_config("experiments/dlm_loss_aggregation/exp004/config.json")
    rows = [
        {"method": method, "correct": 1, "num_examples": 2, "accuracy": 0.5}
        for method in (
            "UNIFORM-ABS", "UNIFORM-SQUARE", "REVEAL-ABS",
            "REVEAL-SQUARE", "REMAIN-ABS", "REMAIN-SQUARE",
        )
    ]
    comparisons = [
        {
            "method_a": left, "method_b": right, "both_correct": 1,
            "a_only_correct": 0, "b_only_correct": 0, "both_wrong": 1,
            "accuracy_difference_pp": 0.0, "p_exact_two_sided": 1.0,
            "primary": index == 2, **({} if index == 2 else {"p_holm": 1.0}),
        }
        for index, (left, right) in enumerate(config["statistics"]["paired_comparisons"])
    ]
    diagnostics = {"rows": [{
        "scope": "global", "pair": "reveal_abs/remain_abs", "spearman": 1.0,
        "mask_xor": 0.0, "mask_iou": 1.0, "topk_overlap": 1.0,
    }]}

    report = _report(config, rows, comparisons, diagnostics, {"passed": True})

    assert "# EXP-004" in report
    assert "## Exact Weighting Equations" in report
    assert "REVEAL-ABS vs REMAIN-ABS" in report
    assert "## Implementation Deviations" in report
