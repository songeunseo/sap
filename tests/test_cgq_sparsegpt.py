import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cgq_sparsegpt import (
    build_corrupted_states,
    cache_dense_references,
    calibration_statistics,
    cgq_token_weights,
    compare_model_masks,
    compare_packed_mask,
    diagonal_change,
    evaluate_heldout,
    masked_logit_sums,
    save_plain_model_masks,
    summarize_mask_cells,
    state_digest,
    write_packed_mask,
)
from lib.prune_llada import prune_sparsegpt


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)

    def forward(self, hidden, attention_bias=None, layer_past=None):
        return (self.q_proj(hidden),)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, hidden_size=2)
        self.seqlen = 4
        self.hf_device_map = {}
        self.embed = nn.Embedding(16, 2)
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.blocks = nn.ModuleList([ToyBlock()])

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for block in self.model.transformer.blocks:
            hidden = block(
                hidden, attention_bias=torch.zeros(1), layer_past=torch.zeros(1)
            )[0]
        return SimpleNamespace(logits=hidden)


class RecordingSparseGPT:
    calls = []

    def __init__(self, layer):
        self.layer = layer
        self.H = torch.zeros(2, 2)

    def add_batch(self, _inputs, _outputs, token_weights=None):
        self.calls.append(token_weights.clone())
        self.H += torch.eye(2) * token_weights.sum()

    def fasterprune(self, *_args, **_kwargs):
        pass

    def free(self):
        pass


def test_sparsegpt_uses_fixed_states_weights_and_records_hessian_diagonal(monkeypatch):
    RecordingSparseGPT.calls = []
    monkeypatch.setattr("lib.prune_llada.SparseGPT", RecordingSparseGPT)
    loader = [
        (torch.tensor([[1, 2, 3, 4]]),),
        (torch.tensor([[5, 6, 7, 8]]),),
    ]
    token_weights = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    ]
    hessian_diagonals = {}
    model = ToyModel()

    prune_sparsegpt(
        SimpleNamespace(nsamples=2, seed=0, sparsity_ratio=0.5),
        model,
        tokenizer=None,
        dev=torch.device("cpu"),
        calibration_loader=loader,
        token_weights=token_weights,
        hessian_diagonals=hessian_diagonals,
    )

    assert len(RecordingSparseGPT.calls) == 2
    torch.testing.assert_close(RecordingSparseGPT.calls[0], token_weights[0])
    torch.testing.assert_close(RecordingSparseGPT.calls[1], token_weights[1])
    torch.testing.assert_close(
        hessian_diagonals["model.transformer.blocks.0.q_proj"],
        torch.tensor([36.0, 36.0]),
    )
    assert model.config.use_cache is True


def test_cgq_token_weights_use_mask_status_and_detached_confidence():
    input_ids = torch.tensor([[99, 7]])
    logits = torch.tensor(
        [[[math.log(4), 0.0], [0.0, math.log(3)]]], requires_grad=True
    )

    weights, confidence, mask_weight = cgq_token_weights(input_ids, logits, mask_id=99)

    torch.testing.assert_close(confidence, torch.tensor([[0.8, 0.75]]))
    torch.testing.assert_close(mask_weight, torch.tensor([[1.0, 0.7]]))
    torch.testing.assert_close(
        weights,
        torch.tensor([[1.0 + math.sqrt(0.8), 0.7 + math.sqrt(0.75)]]),
    )
    assert not weights.requires_grad
    assert torch.isfinite(weights).all()


def test_corrupted_states_and_digest_are_deterministic():
    clean = [torch.arange(12).reshape(1, 12), torch.arange(12, 24).reshape(1, 12)]

    first = build_corrupted_states(clean, (0.2, 0.8), mask_id=99, seed=7)
    second = build_corrupted_states(clean, (0.2, 0.8), mask_id=99, seed=7)

    assert len(first) == 4
    assert state_digest(first) == state_digest(second)
    assert all(
        torch.equal(left["input_ids"], right["input_ids"])
        for left, right in zip(first, second)
    )
    assert all(state["mask"].any() for state in first)


class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        logits = torch.stack((input_ids.float(), -input_ids.float()), dim=-1)
        return SimpleNamespace(logits=logits)


def test_dense_references_cache_only_masked_logits_on_cpu_once():
    states = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "mask": torch.tensor([[True, False, True]]),
            "timestep": 0.2,
        },
        {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "mask": torch.tensor([[False, True, False]]),
            "timestep": 0.8,
        },
    ]
    model = ReferenceModel()

    references = cache_dense_references(model, states, torch.device("cpu"))

    assert model.calls == 2
    assert [reference["logits"].shape for reference in references] == [(2, 2), (1, 2)]
    assert all(reference["logits"].device.type == "cpu" for reference in references)
    assert all(reference["logits"].dtype == torch.float16 for reference in references)


def test_masked_logit_sums_compute_dense_to_sparse_kl_in_fp32():
    dense = torch.log(torch.tensor([[0.75, 0.25], [0.1, 0.9]])).half()
    sparse = torch.log(torch.tensor([[0.5, 0.5], [0.8, 0.2]])).half()

    result = masked_logit_sums(dense, sparse)

    expected_kl = (
        0.75 * math.log(1.5)
        + 0.25 * math.log(0.5)
        + 0.1 * math.log(0.125)
        + 0.9 * math.log(4.5)
    )
    assert result["count"] == 2
    assert result["agreement_count"] == 1
    assert result["kl_sum"] == pytest.approx(expected_kl, abs=2e-3)
    assert result["confidence_abs_error_sum"] == pytest.approx(0.35, abs=2e-3)


def test_diagonal_change_reports_cosine_and_relative_l2():
    result = diagonal_change(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0]))

    assert result["cosine_similarity"] == pytest.approx(0.9486833)
    assert result["relative_l2_difference"] == pytest.approx(0.4472136)


def test_calibration_statistics_split_masked_and_unmasked_tokens_by_timestep():
    states = [
        {"mask": torch.tensor([[True, False]]), "timestep": 0.2},
        {"mask": torch.tensor([[True, True]]), "timestep": 0.8},
    ]
    confidence = [torch.tensor([[0.81, 0.25]]), torch.tensor([[0.36, 0.49]])]
    weights = [torch.tensor([[1.9, 1.2]]), torch.tensor([[1.6, 1.7]])]

    result = calibration_statistics(states, confidence, weights)

    assert result["overall"]["masked_ratio"] == pytest.approx(0.75)
    assert result["overall"]["masked"]["count"] == 3
    assert result["overall"]["masked"]["confidence_mean"] == pytest.approx(
        (0.81 + 0.36 + 0.49) / 3
    )
    assert result["overall"]["unmasked"]["r_mean"] == pytest.approx(1.2)
    assert result["by_timestep"][0]["timestep"] == 0.2
    assert result["by_timestep"][1]["masked_ratio"] == 1.0


def test_packed_masks_report_exact_xor_and_aggregate_by_layer(tmp_path):
    first_plain = torch.tensor([True, False, True, False, True, False, True, False])
    second_plain = torch.tensor([True, True, False, False])
    first_cgq = torch.tensor([True, True, False, False, True, False, True, False])
    second_cgq = torch.tensor([False, True, False, True])
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    write_packed_mask(first_path, first_plain)
    write_packed_mask(second_path, second_plain)

    cells = [
        {
            "block": 0,
            "module": "q_proj",
            **compare_packed_mask(first_path, first_cgq),
        },
        {
            "block": 1,
            "module": "ff_out",
            **compare_packed_mask(second_path, second_cgq),
        },
    ]
    result = summarize_mask_cells(cells)

    assert cells[0]["xor_count"] == 2
    assert cells[1]["xor_count"] == 2
    assert result["global_xor_fraction"] == pytest.approx(4 / 12)
    assert result["mean_layer_xor_fraction"] == pytest.approx((2 / 8 + 2 / 4) / 2)
    assert result["mean_module_xor_fraction"] == pytest.approx((2 / 8 + 2 / 4) / 2)
    assert result["top_cells"][0]["module"] == "ff_out"


def test_model_masks_record_exact_sparsity_and_compare_after_pruning(tmp_path):
    model = ToyModel()
    model.model.transformer.blocks[0].q_proj.weight.data.copy_(
        torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    )

    plain = save_plain_model_masks(model, tmp_path)
    model.model.transformer.blocks[0].q_proj.weight.data.copy_(
        torch.tensor([[0.0, 0.0], [2.0, 3.0]])
    )
    cgq, cells = compare_model_masks(model, tmp_path)

    assert plain == {"zero_count": 2, "total_count": 4, "sparsity": 0.5}
    assert cgq == {"zero_count": 2, "total_count": 4, "sparsity": 0.5}
    assert cells[0]["block"] == 0
    assert cells[0]["module"] == "q_proj"
    assert cells[0]["xor_fraction"] == 0.5


class ScaledReferenceModel(ReferenceModel):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input_ids):
        self.calls += 1
        values = input_ids.float() * self.scale
        return SimpleNamespace(logits=torch.stack((values, -values), dim=-1))


def test_heldout_evaluation_aggregates_the_shared_references_by_timestep():
    states = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "mask": torch.tensor([[True, False, True]]),
            "timestep": 0.2,
        },
        {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "mask": torch.tensor([[False, True, False]]),
            "timestep": 0.8,
        },
    ]
    references = cache_dense_references(
        ScaledReferenceModel(1.0), states, torch.device("cpu")
    )

    result = evaluate_heldout(
        ScaledReferenceModel(0.5), states, references, torch.device("cpu")
    )

    assert result["overall"]["masked_token_count"] == 3
    assert result["overall"]["prediction_agreement"] == 1.0
    assert [row["timestep"] for row in result["by_timestep"]] == [0.2, 0.8]
    assert all(row["kl"] > 0 for row in result["by_timestep"])
