import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    METHOD_ORDER,
    _merge_timing_rows,
    _validated_dense_fingerprint,
    _zero_mask_summary,
    apply_dlm_masks,
    dense_fingerprint,
    extract_gsm8k_records,
    paired_correctness,
    validate_timing_gate,
    validate_exp001_masks,
)


def _write_mask(root, method, mask, block=0, module="linear"):
    payload = pack_mask(mask)
    path = root / method / f"block_{block:02d}__{module}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload["bits"])
    return {
        "method": method,
        "block_index": block,
        "module": module,
        "shape": list(mask.shape),
        "byte_length": len(payload["bits"]),
        "sha256": mask_sha256(payload),
        "runtime_path": str(path),
    }


def _overall(entries, method):
    hasher = hashlib.sha256()
    for entry in entries:
        if entry["method"] == method:
            document = {
                key: entry[key]
                for key in ("block_index", "module", "shape", "sha256")
            }
            hasher.update(
                json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            )
    return hasher.hexdigest()


def test_validate_exp001_masks_rejects_hash_or_row_sparsity_mismatch(tmp_path):
    mask = torch.tensor([[True, True, False, False]])
    entries = [_write_mask(tmp_path, method, mask) for method in ("sum", "abs", "square")]
    metadata = {
        "runtime_directory": str(tmp_path),
        "entries": entries,
        "total_bytes": sum(entry["byte_length"] for entry in entries),
        "overall_sha256": {method: _overall(entries, method) for method in ("sum", "abs", "square")},
    }
    scores = {
        "matrix_count": 1,
        "state_count": 80,
        "modules": [{"layer": 0, "module": "linear", "shape": [1, 4], "update_count": 80}],
    }

    result = validate_exp001_masks(metadata, scores, expected_matrices=1)
    assert result["counts"] == {"sum": 1, "abs": 1, "square": 1}

    Path(entries[0]["runtime_path"]).write_bytes(b"\x00")
    with pytest.raises(ValueError, match="hash"):
        validate_exp001_masks(metadata, scores, expected_matrices=1)


def test_apply_dlm_masks_is_modulewise_and_reproduces_exp001_hashes(tmp_path):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].linear = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.blocks[0].linear.weight.copy_(torch.arange(1, 9).reshape(2, 4))
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    entry = _write_mask(tmp_path, "sum", mask)
    expected = _overall([entry], "sum")

    result = apply_dlm_masks(
        model,
        [entry],
        expected,
        get_modules=lambda candidate: {(0, "linear"): candidate.blocks[0].linear},
    )

    assert result["mask_hash"] == expected
    assert result["sparsity"] == 0.5
    assert torch.equal(model.blocks[0].linear.weight == 0, mask)


def test_extract_gsm8k_records_uses_primary_strict_match_once_per_example():
    samples = [
        {
            "doc_id": 7,
            "doc": {"question": "2+2?", "answer": "work #### 4"},
            "target": "work #### 4",
            "resps": [["reasoning #### 4"]],
            "filtered_resps": ["4"],
            "filter": "strict-match",
            "exact_match": 1.0,
            "doc_hash": "d",
            "prompt_hash": "p",
            "target_hash": "t",
        },
        {
            "doc_id": 7,
            "doc": {"question": "2+2?", "answer": "work #### 4"},
            "target": "work #### 4",
            "resps": [["reasoning #### 4"]],
            "filtered_resps": ["4"],
            "filter": "flexible-extract",
            "exact_match": 1.0,
            "doc_hash": "d",
            "prompt_hash": "p",
            "target_hash": "t",
        },
    ]

    records = extract_gsm8k_records(samples, "DLM-ABS", "cfg")

    assert records == [{
        "example_id": 7,
        "doc_hash": "d",
        "prompt_hash": "p",
        "target_hash": "t",
        "reference_answer": "work #### 4",
        "generated_text": "reasoning #### 4",
        "extracted_answer": "4",
        "correct": True,
        "method": "DLM-ABS",
        "evaluation_config_hash": "cfg",
    }]


def test_paired_correctness_counts_all_four_cells_and_rejects_different_examples():
    left = [
        {"example_id": 0, "doc_hash": "a", "correct": True},
        {"example_id": 1, "doc_hash": "b", "correct": True},
        {"example_id": 2, "doc_hash": "c", "correct": False},
        {"example_id": 3, "doc_hash": "d", "correct": False},
    ]
    right = [
        {"example_id": 0, "doc_hash": "a", "correct": True},
        {"example_id": 1, "doc_hash": "b", "correct": False},
        {"example_id": 2, "doc_hash": "c", "correct": True},
        {"example_id": 3, "doc_hash": "d", "correct": False},
    ]

    assert paired_correctness(left, right) == {
        "both_correct": 1,
        "a_correct_b_wrong": 1,
        "a_wrong_b_correct": 1,
        "both_wrong": 1,
    }
    right[0]["doc_hash"] = "different"
    with pytest.raises(ValueError, match="examples differ"):
        paired_correctness(left, right)


def test_dense_fingerprint_covers_weight_values_and_module_identity():
    first = torch.nn.Linear(3, 2, bias=False)
    second = torch.nn.Linear(3, 2, bias=False)
    second.load_state_dict(first.state_dict())

    baseline = dense_fingerprint({(0, "linear"): first})
    assert baseline == dense_fingerprint({(0, "linear"): second})
    with torch.no_grad():
        second.weight[0, 0] += 1
    assert baseline != dense_fingerprint({(0, "linear"): second})
    assert baseline != dense_fingerprint({(1, "linear"): first})


def test_validated_dense_fingerprint_returns_only_the_digest_not_layer_references():
    model = torch.nn.Module()
    model.config = type("Config", (), {"_commit_hash": "revision"})()
    model.model = torch.nn.Module()
    model.model.transformer = torch.nn.Module()
    block = torch.nn.Module()
    block.linear = torch.nn.Linear(4, 2, bias=False)
    model.model.transformer.blocks = torch.nn.ModuleList([block])
    config = {
        "model": {"revision": "revision"},
        "source_exp001": {"matrix_count": 1},
    }
    scores = {
        "modules": [{"layer": 0, "module": "linear", "shape": [2, 4]}]
    }

    digest = _validated_dense_fingerprint(model, config, scores)

    assert isinstance(digest, str)
    assert len(digest) == 64


def test_timing_gate_requires_every_method_and_rejects_unsafe_or_pathological_runs():
    methods = ["Dense", "DLM-SUM", "DLM-ABS"]
    rows = [
        {"method": method, "examples_per_second": speed, "peak_cuda_reserved_bytes": 10}
        for method, speed in zip(methods, (1.0, 0.9, 0.8))
    ]
    assert validate_timing_gate(rows, methods, max_cuda_bytes=20, slowdown_factor=3) == {
        "passed": True,
        "slowest_to_fastest_ratio": 1.25,
    }

    with pytest.raises(ValueError, match="missing"):
        validate_timing_gate(rows[:-1], methods, max_cuda_bytes=20, slowdown_factor=3)
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_timing_gate(rows, methods, max_cuda_bytes=5, slowdown_factor=3)
    rows[-1]["examples_per_second"] = 0.2
    with pytest.raises(RuntimeError, match="pathological"):
        validate_timing_gate(rows, methods, max_cuda_bytes=20, slowdown_factor=3)


def test_standard_sparsegpt_accepts_global_half_without_claiming_rowwise_half():
    layer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 4.0]])
        )
    modules = {(0, "linear"): layer}

    result = _zero_mask_summary(
        object(),
        "sparsegpt",
        require_exact_rowwise=False,
        get_modules=lambda _: modules,
    )

    assert result["sparsity"] == 0.5
    assert result["rowwise_exact"] is False
    assert result["row_prune_min"] == 1
    assert result["row_prune_max"] == 3
    with pytest.raises(ValueError, match="row sparsity"):
        _zero_mask_summary(
            object(),
            "wanda",
            require_exact_rowwise=True,
            get_modules=lambda _: modules,
        )


def test_resume_timing_accepts_only_the_missing_suffix_in_frozen_order():
    existing = [{"method": method} for method in METHOD_ORDER[:-1]]
    resumed = [{"method": "SparseGPT"}]

    assert _merge_timing_rows(existing, resumed) == existing + resumed

    with pytest.raises(ValueError, match="prefix"):
        _merge_timing_rows(existing[1:], resumed)
