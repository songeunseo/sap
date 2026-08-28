import json

import pytest
import torch

from cgq_sparsegpt import state_digest
from cgq_sparsegpt_downstream import (
    assert_same_examples,
    extract_winogrande_records,
    load_cached_calibration_states,
    make_in_memory_harness,
    summarize_paired_records,
    validate_mask_reproduction,
)


def _sample(doc_id, scores, correct):
    return {
        "doc_id": doc_id,
        "doc": {"answer": "2"},
        "doc_hash": f"doc-{doc_id}",
        "prompt_hash": f"prompt-{doc_id}",
        "target_hash": f"target-{doc_id}",
        "filtered_resps": [[str(scores[0]), "0.0"], [str(scores[1]), "0.0"]],
        "acc": float(correct),
    }


def test_in_memory_harness_preserves_established_winogrande_settings():
    model = object()
    tokenizer = object()

    harness = make_in_memory_harness(model, tokenizer, torch.device("cuda:0"))

    assert harness.model is model
    assert harness.tokenizer is tokenizer
    assert harness.device == torch.device("cuda:0")
    assert harness.mask_id == 126336
    assert harness.batch_size == 8
    assert harness.mc_num == 128
    assert harness.cfg == 0.0
    assert harness.is_check_greedy is False
    assert harness.rank == 0
    assert harness.world_size == 1


def test_winogrande_records_and_pairing_use_real_scores_and_correctness():
    plain = extract_winogrande_records(
        [_sample(0, (-2.0, -1.0), True), _sample(1, (-1.0, -2.0), False)]
    )
    cgq = extract_winogrande_records(
        [_sample(0, (-1.0, -2.0), False), _sample(1, (-2.0, -1.0), True)]
    )

    assert plain[0]["prediction_index"] == 1
    assert plain[0]["target_index"] == 1
    assert plain[0]["correct"] is True
    assert summarize_paired_records(plain, cgq) == {
        "sample_count": 2,
        "plain_correct": 1,
        "cgq_correct": 1,
        "plain_accuracy": 0.5,
        "cgq_accuracy": 0.5,
        "delta_percentage_points": 0.0,
        "plain_wrong_cgq_correct": 1,
        "plain_correct_cgq_wrong": 1,
        "both_correct": 0,
        "both_wrong": 0,
    }


def test_example_validation_rejects_a_different_prompt():
    reference = extract_winogrande_records([_sample(0, (-2.0, -1.0), True)])
    candidate = [dict(reference[0], prompt_hash="different")]

    with pytest.raises(ValueError, match="prompt_hash"):
        assert_same_examples(reference, candidate)


def test_cached_calibration_must_match_previous_experiment_digest(tmp_path):
    states = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "mask": torch.tensor([[False, True, False]]),
            "timestep": 0.2,
        }
    ]
    state_path = tmp_path / "states.pt"
    result_path = tmp_path / "results.json"
    torch.save({"calibration": states}, state_path)
    result_path.write_text(
        json.dumps({"calibration_state_digest": state_digest(states)})
    )

    loaded, digest = load_cached_calibration_states(state_path, result_path)

    assert digest == state_digest(states)
    torch.testing.assert_close(loaded[0]["input_ids"], states[0]["input_ids"])


def test_mask_reproduction_stops_a_materially_different_run():
    validate_mask_reproduction(0.081, expected=0.080565, tolerance=0.005)

    with pytest.raises(RuntimeError, match="mask XOR"):
        validate_mask_reproduction(0.070, expected=0.080565, tolerance=0.005)
