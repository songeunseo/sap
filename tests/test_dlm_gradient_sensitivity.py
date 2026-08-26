import copy
import hashlib
import json
import math
import weakref
from pathlib import Path

import pytest
import torch
import dlm_gradient_sensitivity as sensitivity_cli
import lib.dlm_gradient_sensitivity as sensitivity_lib

from lib.dlm_gradient_sensitivity import (
    NegativeSuffixCostError,
    TimestepSensitivityAccumulator,
    block_state_gradients,
    embed_state,
    feasibility_gate,
    project_scoring_seconds,
    row_change_fraction,
    suffix_logits,
    top_fraction_overlap,
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
    score_block,
    unpack_mask,
)
from lib.prune_llada import find_layers
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM


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


# Mutation caught: accepting bool as version 1 permits malformed masks or blocks.
def test_mask_and_block_artifact_versions_reject_bool(tmp_path):
    packed = pack_mask(torch.tensor([[True, False, True]]))
    bool_version = dict(packed, version=True)
    header = {
        key: bool_version[key]
        for key in ("version", "shape", "dtype", "bitorder", "prune_count", "per_row_prune_count", "byte_length")
    }
    bool_version["sha256"] = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + bool_version["bits"]
    ).hexdigest()
    with pytest.raises(ValueError, match="header"):
        unpack_mask(bool_version)

    path = tmp_path / "masks.json"
    save_mask_block(path, {"module": torch.tensor([[True]])}, {})
    document = json.loads(path.read_text())
    document["version"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="block"):
        load_mask_block(path)


# Mutation caught: failing after tempfile creation must not replace or leak beside a valid artifact.
def test_save_mask_block_keeps_existing_artifact_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "masks.json"
    original = {"module": torch.tensor([[True, False, True]])}
    save_mask_block(path, original, {"run": "old"})
    original_bytes = path.read_bytes()
    temporary_paths = []

    def fail_replace(self, target):
        temporary_paths.append(self)
        raise OSError("interrupted before replace")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_mask_block(path, {"module": torch.tensor([[False, True, False]])}, {"run": "new"})
    assert temporary_paths and not temporary_paths[0].exists()
    assert path.read_bytes() == original_bytes


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
def test_jaccard_captures_mask_diagnostics():
    left = torch.tensor([[True, True, False, False], [True, False, True, False]])
    right = torch.tensor([[True, False, True, False], [True, False, True, False]])
    assert jaccard(left, right) == pytest.approx(0.6)


# Mutation caught: counting changed weights rather than rows misstates selection instability.
def test_row_change_fraction_counts_rows_and_rejects_invalid_masks():
    left = torch.tensor([[True, True, False, False], [True, False, True, False]])
    right = torch.tensor([[True, False, True, False], [True, False, True, False]])
    assert row_change_fraction(left, right) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="two-dimensional"):
        row_change_fraction(torch.tensor([True]), torch.tensor([True]))
    with pytest.raises(ValueError, match="nonempty"):
        row_change_fraction(torch.empty(0, 2, dtype=torch.bool), torch.empty(0, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        row_change_fraction(torch.ones(1, 1), torch.ones(1, 1))


# Mutation caught: taking a global value threshold instead of the deterministic top fraction changes overlap.
def test_top_fraction_overlap_uses_top_one_percent_and_rejects_invalid_inputs():
    sigma_a = torch.arange(100)
    sigma_b = torch.arange(99, -1, -1)
    assert top_fraction_overlap(sigma_a, sigma_b, 0.01) == 0.0
    assert top_fraction_overlap(sigma_a, sigma_a, 0.01) == 1.0
    assert top_fraction_overlap(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]), 0.01) == 1.0
    with pytest.raises(ValueError, match="nonempty"):
        top_fraction_overlap(torch.tensor([]), torch.tensor([]), 0.01)
    with pytest.raises(ValueError, match="fraction"):
        top_fraction_overlap(torch.ones(2), torch.ones(2), 0.0)
    with pytest.raises(ValueError, match="matching"):
        top_fraction_overlap(torch.ones(2), torch.ones(3), 0.01)


def _tiny_llada(n_layers=2, d_model=8, vocab_size=19):
    config = LLaDAConfig(
        d_model=d_model,
        n_heads=1,
        n_layers=n_layers,
        mlp_hidden_size=d_model * 2,
        activation_type="silu",
        block_type="llama",
        rope=True,
        rope_full_precision=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        input_emb_norm=True,
        max_sequence_length=8,
        vocab_size=vocab_size,
        embedding_size=None,
        weight_tying=True,
        scale_logits=True,
        mask_token_id=vocab_size - 1,
        use_cache=False,
    )
    return LLaDAModelLM(config, init_params=True).eval()


# Mutation caught: skipping embedding scaling, final norm, tied head, or logit scaling breaks parity.
def test_suffix_logits_matches_real_llada_forward_from_embedding_and_block_cache():
    torch.manual_seed(0)
    model = _tiny_llada()
    noisy_ids = torch.tensor([[1, 2, 3, 4]])

    full = model(noisy_ids).logits
    cached = embed_state(model, noisy_ids)
    torch.testing.assert_close(suffix_logits(model, cached, start_block=0), full)

    block_zero_output, _ = model.model.transformer.blocks[0](
        cached, attention_bias=None, layer_past=None, use_cache=False
    )
    torch.testing.assert_close(
        suffix_logits(model, block_zero_output, start_block=1), full
    )


# Mutation caught: detaching the frozen suffix or selecting non-block weights changes/loses gradients.
def test_block_state_gradients_match_full_forward_and_do_not_mutate_parameters():
    torch.manual_seed(1)
    model = _tiny_llada()
    target = find_layers(model.model.transformer.blocks[0])
    assert len(target) == 7
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in target.values():
        layer.weight.requires_grad_(True)

    clean_ids = torch.tensor([[1, 2, 3, 4]])
    noisy_ids = torch.tensor([[18, 2, 18, 4]])
    mask = torch.tensor([[True, False, True, False]])
    hidden = embed_state(model, noisy_ids)
    parameter_bytes = {
        name: parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters()
    }

    official_dlm_loss(model(noisy_ids).logits, clean_ids, mask, 0.5).backward()
    expected = {name: layer.weight.grad.detach().clone() for name, layer in target.items()}
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("model.transformer.blocks.0.") or ".weight" not in name
    )
    model.zero_grad(set_to_none=True)
    original_flags = tuple(parameter.requires_grad for parameter in model.parameters())

    actual, next_cache = block_state_gradients(
        model, 0, hidden, clean_ids, mask, p_mask=0.5
    )

    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name].float().cpu())
        assert torch.isfinite(actual[name]).all()
    assert not next_cache.requires_grad
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == original_flags
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {
        name: parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters()
    } == parameter_bytes


# Mutation caught: restoring flags only on success leaks trainable parameters after a failed state.
def test_block_state_gradients_restores_flags_and_bytes_on_failure():
    model = _tiny_llada()
    parameters = list(model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.requires_grad_(index % 3 == 0)
    flags = tuple(parameter.requires_grad for parameter in parameters)
    parameter_bytes = [parameter.detach().cpu().numpy().tobytes() for parameter in parameters]

    clean_ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="masked token"):
        block_state_gradients(
            model,
            0,
            embed_state(model, clean_ids),
            clean_ids,
            torch.zeros_like(clean_ids, dtype=torch.bool),
            p_mask=0.5,
        )

    assert tuple(parameter.requires_grad for parameter in parameters) == flags
    assert [parameter.detach().cpu().numpy().tobytes() for parameter in parameters] == parameter_bytes
    assert all(parameter.grad is None for parameter in parameters)


# Mutation caught: charging all blocks as block 0 or omitting 80 states changes 44,800 seconds.
def test_project_scoring_seconds_fits_suffix_cost_and_rejects_negative_slope():
    projection = project_scoring_seconds(2.0, 33.0)
    assert projection["fixed_seconds"] == pytest.approx(2.0)
    assert projection["suffix_seconds"] == pytest.approx(1.0)
    assert projection["projected_scoring_seconds"] == pytest.approx(44_800.0)
    with pytest.raises(NegativeSuffixCostError, match="suffix"):
        project_scoring_seconds(2.0, 1.0)


@pytest.mark.parametrize(
    ("last_seconds", "first_seconds", "blocks", "states"),
    [
        (1.0, -1.0, 32, 80),
        (float("inf"), 1.0, 32, 80),
        (float("nan"), 1.0, 32, 80),
        (2.0, 1.0, 1, 80),
        (2.0, 1.0, 32, 0),
    ],
)
def test_stage0_projection_does_not_constrain_invalid_projector_inputs(
    last_seconds, first_seconds, blocks, states
):
    with pytest.raises(ValueError) as error:
        sensitivity_cli._project_stage0_seconds(
            last_seconds, first_seconds, blocks, states
        )
    assert not isinstance(error.value, NegativeSuffixCostError)


# Mutation caught: strict inequalities reject scientifically valid exact-boundary runs.
def test_feasibility_gate_accepts_exact_memory_and_runtime_boundaries():
    gib = 1024**3
    measurements = [
        {
            "max_memory_allocated_bytes": 30 * gib,
            "max_memory_reserved_bytes": 30 * gib,
            "vm_swap_delta_kib": 0,
            "gradient_element_count": 7,
            "finite_gradient_count": 7,
            "nonzero_gradient_count": 1,
        }
        for _ in range(2)
    ]
    decision = feasibility_gate(measurements, projected_seconds=24 * 60 * 60)
    assert decision["passed"]
    assert all(decision["checks"].values())
    assert len(decision["measurement_checks"]) == 2
    assert all(all(check["checks"].values()) for check in decision["measurement_checks"])

    measurements[0]["max_memory_reserved_bytes"] += 1
    assert not feasibility_gate(measurements, 24 * 60 * 60)["passed"]
    measurements[0]["max_memory_reserved_bytes"] -= 1
    measurements[1]["vm_swap_delta_kib"] = 1
    assert not feasibility_gate(measurements, 24 * 60 * 60)["passed"]
    measurements[1]["vm_swap_delta_kib"] = 0
    assert not feasibility_gate(measurements, math.nextafter(24 * 60 * 60, math.inf))["passed"]


# Mutation caught: returning before serialization loses scientifically valid NO-GO evidence.
def test_feasibility_cli_writes_gate_failure_and_returns_two(tmp_path, monkeypatch):
    config_path = tmp_path / "pilot.json"
    output_path = tmp_path / "stage0.json"
    config_path.write_text("{}")
    report = {"stage": 0, "gate": {"passed": False, "checks": {"runtime": False}}}
    monkeypatch.setattr(sensitivity_cli, "run_feasibility", lambda config: report)

    status = sensitivity_cli.main(
        ["feasibility", "--config", str(config_path), "--output", str(output_path)]
    )

    assert status == 2
    assert json.loads(output_path.read_text()) == report


# Mutation caught: reading only the current CUDA device can hide the true peak on a mapped block.
def test_cuda_peak_accounting_covers_every_model_device_and_rejects_offload(monkeypatch):
    reset_devices = []
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: reset_devices.append(device)
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: {0: 11, 2: 31}[device],
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: {0: 13, 2: 37}[device],
    )

    sensitivity_cli._reset_cuda_peaks((0, 2))
    peaks = sensitivity_cli._read_cuda_peaks((0, 2))

    assert reset_devices == [0, 2]
    assert peaks == {
        "devices": [
            {
                "device": "cuda:0",
                "max_memory_allocated_bytes": 11,
                "max_memory_reserved_bytes": 13,
            },
            {
                "device": "cuda:2",
                "max_memory_allocated_bytes": 31,
                "max_memory_reserved_bytes": 37,
            },
        ],
        "max_memory_allocated_bytes": 31,
        "max_memory_reserved_bytes": 37,
    }

    offloaded = type(
        "OffloadedModel", (), {"hf_device_map": {"model.embed": "cuda:0", "model.block": "cpu"}}
    )()
    with pytest.raises(RuntimeError, match="offload"):
        sensitivity_cli._model_cuda_devices(offloaded)


# Mutation caught: moving replay/finish into the state timer or omitting finalization changes projection.
@pytest.mark.parametrize(
    ("clock_values", "expected_state", "expected_total", "fit_status", "fit_method", "raw_difference", "raw_suffix"),
    [
        ((0.0, 2.0, 10.0, 10.5, 20.0, 53.0, 60.0, 60.25), 44_800, 44_960, "unconstrained", "two_point_linear", 31.0, 1.0),
        (
            (0.0, 0.503889, 10.0, 10.5, 20.0, 20.474458, 60.0, 60.25),
            80 * 32 * 0.503889,
            80 * 32 * 0.503889 + 160,
            "constrained",
            "max_endpoint_constant",
            0.474458 - 0.503889,
            (0.474458 - 0.503889) / 31,
        ),
    ],
)
def test_loaded_feasibility_runs_real_tiny_block_workflow_with_exact_timing(
    monkeypatch,
    clock_values,
    expected_state,
    expected_total,
    fit_status,
    fit_method,
    raw_difference,
    raw_suffix,
):
    torch.manual_seed(2)
    model = _tiny_llada(n_layers=32, d_model=4, vocab_size=7)
    clean_ids = torch.tensor([[1, 2]])
    config = {
        "model": {"id": "tiny", "revision": "test"},
        "dataset": {"loader_name": "tiny"},
        "calibration": {
            "seed": 0,
            "sequence_count": 8,
            "sequence_length": 2,
            "timesteps": list(midpoint_timesteps()),
        },
        "stage0": {
            "model_block_count": 32,
            "measured_blocks": [31, 0],
            "max_gpu_memory_gib": 30,
            "max_projected_gpu_hours": 24,
        },
    }
    events = []
    clock_values = iter(clock_values)

    def clock():
        value = next(clock_values)
        events.append(("clock", value))
        return value

    real_gradients = sensitivity_cli.block_state_gradients

    def tracked_gradients(*args, **kwargs):
        gradients, cache = real_gradients(*args, **kwargs)
        events.append(("backward", args[1], all(g.dtype == torch.float32 for g in gradients.values())))
        return gradients, cache

    real_accumulator = sensitivity_cli.TimestepSensitivityAccumulator

    class TrackedAccumulator(real_accumulator):
        def add_state(self, gradients, split):
            super().add_state(gradients, split)
            squared = all(
                torch.equal(self._sum_sq[split][name], gradient.square())
                for name, gradient in gradients.items()
            ) if self._count[split] == 1 else True
            events.append(("add", split, all(g.dtype == torch.float32 for g in gradients.values()), squared))

        def finish_timestep(self):
            events.append(("finish",))
            super().finish_timestep()

    monkeypatch.setattr(sensitivity_cli, "block_state_gradients", tracked_gradients)
    monkeypatch.setattr(sensitivity_cli, "TimestepSensitivityAccumulator", TrackedAccumulator)
    monkeypatch.setattr(sensitivity_cli, "_synchronize", lambda devices=(): None)
    monkeypatch.setattr(
        sensitivity_cli,
        "_process_memory",
        lambda: {"vm_rss_kib": 100, "vm_swap_kib": 0},
    )

    report = sensitivity_cli._run_loaded_feasibility(
        model, config, clean_ids, cuda_devices=(), clock=clock
    )

    assert [event[1] for event in events if event[0] == "backward"] == [31, 0]
    assert all(event[2] for event in events if event[0] == "backward")
    adds = [event for event in events if event[0] == "add"]
    assert len(adds) == 16
    assert all(event[2] and event[3] for event in adds)
    for backward_position in [
        index for index, event in enumerate(events) if event[0] == "backward"
    ]:
        state_end = next(
            index for index in range(backward_position, len(events))
            if events[index][0] == "clock"
        )
        assert [event[0] for event in events[backward_position + 1:state_end]] == ["add"]
        finish_position = next(
            index for index in range(state_end, len(events)) if events[index][0] == "finish"
        )
        assert sum(event[0] == "add" for event in events[state_end + 1:finish_position]) == 7
        assert events[finish_position - 1][0] == "clock"
        assert events[finish_position + 1][0] == "clock"
    assert report["projection"]["fit_status"] == fit_status
    assert report["projection"]["fit_method"] == fit_method
    assert report["projection"]["raw_endpoint_difference_seconds"] == pytest.approx(raw_difference)
    assert report["projection"]["raw_unconstrained_suffix_seconds"] == pytest.approx(raw_suffix)
    assert report["projection"]["projected_scoring_seconds"] == pytest.approx(expected_state)
    assert report["projection"]["projected_statistic_finalization_seconds"] == pytest.approx(160)
    assert report["projection"]["projected_total_seconds"] == pytest.approx(expected_total)
    if fit_status == "constrained":
        assert report["projection"]["fixed_seconds"] == pytest.approx(0.503889)
        assert report["projection"]["suffix_seconds"] == 0
        assert report["projection"]["per_block_seconds"] == pytest.approx([0.503889] * 32)
    else:
        assert report["projection"]["fixed_seconds"] == 2
        assert report["projection"]["suffix_seconds"] == 1
        assert report["projection"]["per_block_seconds"] == [2 + 31 - block for block in range(32)]


def _toy_score_config(blocks=1, timesteps=(0.25, 0.75)):
    return {
        "model": {"id": "tiny", "revision": "test-revision"},
        "dataset": {"loader_name": "tiny"},
        "calibration": {
            "seed": 3,
            "sequence_indices": list(range(8)),
            "sequence_count": 8,
            "sequence_length": 3,
            "timesteps": list(timesteps),
            "split_a": [0, 1, 2, 3],
            "split_b": [4, 5, 6, 7],
        },
        "scoring": {
            "lambdas": [0.25, 0.5, 1.0],
            "sparsities": [0.5, 0.6, 0.7, 0.75],
            "decision_sparsities": [0.5, 0.6, 0.7],
        },
        "stage0": {"model_block_count": blocks},
        "reliability": {
            "spearman_sample_size": 32,
            "minimum_median_split_sigma_spearman": 0.5,
            "minimum_mean_split_mask_jaccard": 0.95,
            "top_sigma_fraction": 0.01,
        },
    }


def _toy_clean_ids():
    return [torch.tensor([[1 + index % 3, 4 + index % 3, 7 + index % 3]]) for index in range(8)]


def _toy_cached_states(model, config):
    states = {}
    calibration = config["calibration"]
    for timestep_index, timestep in enumerate(calibration["timesteps"]):
        for sequence_index, clean_ids in enumerate(_toy_clean_ids()):
            mask_seed = calibration["seed"] + timestep_index * 8 + sequence_index
            noisy_ids, mask, p_mask = make_masked_state(
                clean_ids, timestep, model.config.mask_token_id, mask_seed
            )
            states[(timestep_index, sequence_index)] = {
                "timestep_index": timestep_index,
                "sequence_index": sequence_index,
                "clean_ids": clean_ids,
                "noisy_ids": noisy_ids,
                "mask": mask,
                "p_mask": p_mask,
                "hidden": embed_state(model, noisy_ids).detach().cpu(),
            }
    return states


# Mutation caught: updating Welford per sequence measures sample+timestep dispersion, not timestep dispersion.
def test_score_block_matches_literal_direct_gradient_loop_and_masks(monkeypatch):
    torch.manual_seed(11)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    config = _toy_score_config()
    config["reliability"]["spearman_sample_size"] = 10_000
    states = _toy_cached_states(model, config)
    layers = find_layers(model.model.transformer.blocks[0])
    saved_gradients = {}
    for key in sorted(states):
        state = states[key]
        saved_gradients[key], _ = block_state_gradients(
            model,
            0,
            state["hidden"],
            state["clean_ids"],
            state["mask"],
            state["p_mask"],
        )

    expected = {}
    for name, layer in layers.items():
        timestep_full = []
        timestep_a = []
        timestep_b = []
        weight_sq = layer.weight.detach().float().cpu().square()
        for timestep_index in range(2):
            a = weight_sq * torch.stack(
                [saved_gradients[(timestep_index, index)][name].square() for index in range(4)]
            ).mean(dim=0)
            b = weight_sq * torch.stack(
                [saved_gradients[(timestep_index, index)][name].square() for index in range(4, 8)]
            ).mean(dim=0)
            timestep_a.append(a)
            timestep_b.append(b)
            timestep_full.append((a + b) / 2)
        expected[name] = {
            "mu": torch.stack(timestep_full).mean(dim=0),
            "sigma": torch.stack(timestep_full).std(dim=0, unbiased=False),
            "mu_A": torch.stack(timestep_a).mean(dim=0),
            "sigma_A": torch.stack(timestep_a).std(dim=0, unbiased=False),
            "mu_B": torch.stack(timestep_b).mean(dim=0),
            "sigma_B": torch.stack(timestep_b).std(dim=0, unbiased=False),
        }

    real_gradients = sensitivity_lib.block_state_gradients
    returned_outputs = {}
    call_keys = iter(sorted(states))

    def tracked_gradients(*args, **kwargs):
        key = next(call_keys)
        gradients, output = real_gradients(*args, **kwargs)
        returned_outputs[key] = output.detach().cpu()
        return gradients, output

    block_forward_calls = []
    backward_calls = []
    real_backward = torch.Tensor.backward

    def tracked_backward(tensor, *args, **kwargs):
        backward_calls.append(1)
        return real_backward(tensor, *args, **kwargs)

    hook = model.model.transformer.blocks[0].register_forward_hook(
        lambda *_: block_forward_calls.append(1)
    )
    monkeypatch.setattr(sensitivity_lib, "block_state_gradients", tracked_gradients)
    monkeypatch.setattr(torch.Tensor, "backward", tracked_backward)
    try:
        result, next_states = score_block(model, 0, states, config)
    finally:
        hook.remove()

    def literal_spearman(left, right):
        def ranks(values):
            values = [float(value) for value in values.reshape(-1)]
            ordered = sorted(values)
            return torch.tensor(
                [
                    (
                        ordered.index(value)
                        + len(ordered)
                        - 1
                        - ordered[::-1].index(value)
                        + 2
                    )
                    / 2
                    for value in values
                ],
                dtype=torch.float64,
            )

        left_ranks = ranks(left)
        right_ranks = ranks(right)
        left_ranks -= left_ranks.mean()
        right_ranks -= right_ranks.mean()
        denominator = torch.sqrt(
            left_ranks.square().sum() * right_ranks.square().sum()
        )
        if denominator == 0:
            return None
        return ((left_ranks * right_ranks).sum() / denominator).item()

    def literal_top_overlap(left, right):
        count = max(1, math.ceil(left.numel() * 0.01))
        left_top = set(
            torch.argsort(left.reshape(-1), descending=True, stable=True)[:count].tolist()
        )
        right_top = set(
            torch.argsort(right.reshape(-1), descending=True, stable=True)[:count].tolist()
        )
        return len(left_top & right_top) / len(left_top | right_top)

    def literal_rowwise_mask(score, sparsity):
        mask = torch.zeros_like(score, dtype=torch.bool)
        prune_count = math.floor(score.shape[1] * sparsity)
        for row_index, row in enumerate(score):
            order = sorted(
                range(row.numel()), key=lambda column: (float(row[column]), column)
            )
            mask[row_index, order[:prune_count]] = True
        return mask

    assert result["update_count"] == 2
    assert list(next_states) == list(states)
    assert len(returned_outputs) == len(states) == 16
    assert len(backward_calls) == len(states)
    assert len(block_forward_calls) == len(states)
    for key, output in returned_outputs.items():
        torch.testing.assert_close(next_states[key]["hidden"], output)
    expected_split_rhos = []
    expected_lambda_one_jaccards = {sparsity: [] for sparsity in (0.5, 0.6, 0.7)}
    all_useful_masks_identical = True
    for name, statistics in expected.items():
        for statistic in ("mu", "sigma", "sigma_A", "sigma_B"):
            torch.testing.assert_close(result["statistics"][name][statistic], statistics[statistic])
        for risk_lambda in (0.0, 0.25, 0.5, 1.0):
            score = statistics["mu"] + risk_lambda * statistics["sigma"]
            for sparsity in (0.5, 0.6, 0.7, 0.75):
                assert torch.equal(
                    result["masks"][(name, risk_lambda, sparsity)],
                    literal_rowwise_mask(score, sparsity),
                )
                split_a_mask = literal_rowwise_mask(
                    statistics["mu_A"] + risk_lambda * statistics["sigma_A"], sparsity
                )
                split_b_mask = literal_rowwise_mask(
                    statistics["mu_B"] + risk_lambda * statistics["sigma_B"], sparsity
                )
                diagnostic = result["diagnostics"][name]["masks"][
                    f"lambda={risk_lambda:g}|sparsity={sparsity:g}"
                ]
                assert diagnostic["split_A_sha256"] == pack_mask(split_a_mask)["sha256"]
                assert diagnostic["split_B_sha256"] == pack_mask(split_b_mask)["sha256"]
                intersection = torch.logical_and(split_a_mask, split_b_mask).sum().item()
                union = torch.logical_or(split_a_mask, split_b_mask).sum().item()
                assert diagnostic["split_jaccard"] == pytest.approx(
                    1.0 if union == 0 else intersection / union
                )
                mean_mask = literal_rowwise_mask(statistics["mu"], sparsity)
                changed = result["masks"][(name, risk_lambda, sparsity)].ne(mean_mask)
                assert diagnostic["mean_disagreement_count"] == changed.sum().item()
                assert diagnostic["mean_disagreement_fraction"] == pytest.approx(
                    changed.float().mean().item()
                )
                assert diagnostic["changed_row_fraction"] == pytest.approx(
                    changed.any(dim=1).float().mean().item()
                )
                full_mask = result["masks"][(name, risk_lambda, sparsity)]
                mean_union = torch.logical_or(mean_mask, full_mask).sum().item()
                mean_intersection = torch.logical_and(mean_mask, full_mask).sum().item()
                assert diagnostic["mean_jaccard"] == pytest.approx(
                    1.0 if mean_union == 0 else mean_intersection / mean_union
                )
                expected_identical = not changed.any().item()
                assert diagnostic["identical_to_mean"] is expected_identical
                if risk_lambda and sparsity in expected_lambda_one_jaccards:
                    all_useful_masks_identical &= expected_identical
                if risk_lambda == 1.0 and sparsity in expected_lambda_one_jaccards:
                    expected_lambda_one_jaccards[sparsity].append(
                        1.0 if union == 0 else intersection / union
                    )
        ratio = statistics["sigma"] / (
            statistics["mu"] + torch.finfo(torch.float32).eps
        )
        quantiles = torch.quantile(ratio.flatten(), torch.tensor([0.5, 0.9, 0.99]))
        module_diagnostic = result["diagnostics"][name]
        ratio_diagnostic = module_diagnostic["sigma_over_mu_plus_eps"]
        assert ratio_diagnostic["epsilon"] == torch.finfo(torch.float32).eps
        assert ratio_diagnostic["median"] == pytest.approx(quantiles[0].item())
        assert ratio_diagnostic["p90"] == pytest.approx(quantiles[1].item())
        assert ratio_diagnostic["p99"] == pytest.approx(quantiles[2].item())
        assert ratio_diagnostic["sample_size"] == ratio.numel()
        assert ratio_diagnostic["requested_sample_size"] == 10_000
        expected_sample_digest = hashlib.sha256(
            torch.arange(ratio.numel()).numpy().tobytes()
        ).hexdigest()
        assert ratio_diagnostic["sample_indices_sha256"] == expected_sample_digest
        assert (
            ratio_diagnostic["sample_indices_sha256"]
            == module_diagnostic["rho_mu_sigma"]["sample_indices_sha256"]
        )
        expected_mu_sigma_rho = literal_spearman(statistics["mu"], statistics["sigma"])
        expected_split_rho = literal_spearman(statistics["sigma_A"], statistics["sigma_B"])
        expected_split_rhos.append(expected_split_rho)
        if expected_mu_sigma_rho is None:
            assert module_diagnostic["rho_mu_sigma"]["value"] is None
        else:
            assert module_diagnostic["rho_mu_sigma"]["value"] == pytest.approx(
                expected_mu_sigma_rho
            )
        if expected_split_rho is None:
            assert module_diagnostic["rho_sigma_A_sigma_B"]["value"] is None
        else:
            assert module_diagnostic["rho_sigma_A_sigma_B"]["value"] == pytest.approx(
                expected_split_rho
            )
        assert module_diagnostic["top_sigma_overlap"] == pytest.approx(
            literal_top_overlap(statistics["sigma_A"], statistics["sigma_B"])
        )

    aggregated = sensitivity_cli._stage1_gate(
        list(result["diagnostics"].values()), config, all_useful_masks_identical
    )
    defined_rhos = sorted(value for value in expected_split_rhos if value is not None)
    middle = len(defined_rhos) // 2
    expected_median = (
        defined_rhos[middle]
        if len(defined_rhos) % 2
        else (defined_rhos[middle - 1] + defined_rhos[middle]) / 2
    )
    assert aggregated["median_module_split_sigma_spearman"] == pytest.approx(
        expected_median
    )
    assert aggregated["mean_split_mask_jaccard"] == pytest.approx(
        {
            f"{sparsity:g}": sum(values) / len(values)
            for sparsity, values in expected_lambda_one_jaccards.items()
        }
    )
    assert aggregated["all_useful_masks_identical"] is all_useful_masks_identical


def test_score_block_ratio_quantiles_reuse_bounded_spearman_sample(monkeypatch):
    torch.manual_seed(13)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    config = _toy_score_config(timesteps=(0.5,))
    config["reliability"]["spearman_sample_size"] = 3
    states = _toy_cached_states(model, config)
    quantile_inputs = []
    real_quantile = torch.quantile

    def track_quantile(values, *args, **kwargs):
        quantile_inputs.append(values.detach().clone())
        return real_quantile(values, *args, **kwargs)

    monkeypatch.setattr(torch, "quantile", track_quantile)
    result, _ = score_block(model, 0, states, config)

    assert len(quantile_inputs) == 7
    for (name, statistics), sampled_ratio in zip(
        result["statistics"].items(), quantile_inputs
    ):
        ratio = statistics["sigma"] / (
            statistics["mu"] + torch.finfo(torch.float32).eps
        )
        assert ratio.numel() > 3
        expected_indices = torch.randperm(
            ratio.numel(), generator=torch.Generator().manual_seed(3)
        )[:3]
        torch.testing.assert_close(sampled_ratio, ratio.reshape(-1)[expected_indices])
        expected_digest = hashlib.sha256(expected_indices.numpy().tobytes()).hexdigest()
        diagnostic = result["diagnostics"][name]
        assert diagnostic["sigma_over_mu_plus_eps"]["sample_size"] == 3
        assert diagnostic["sigma_over_mu_plus_eps"]["requested_sample_size"] == 3
        assert (
            diagnostic["sigma_over_mu_plus_eps"]["sample_indices_sha256"]
            == expected_digest
            == diagnostic["rho_mu_sigma"]["sample_indices_sha256"]
        )


# Mutation caught: resume that trusts filenames, silently rescores block 0, or rewrites JSON changes output bytes.
def test_score_resume_skips_verified_block_and_is_byte_identical(tmp_path, monkeypatch):
    config = _toy_score_config(blocks=2, timesteps=(0.5,))
    clean_ids = _toy_clean_ids()
    resumed_dir = tmp_path / "resumed"
    uninterrupted_dir = tmp_path / "uninterrupted"

    torch.manual_seed(19)
    interrupted_model = _tiny_llada(n_layers=2, d_model=4, vocab_size=11)
    real_score_block = sensitivity_cli.score_block

    def interrupt_after_block_zero(model, block_index, cached_states, score_config):
        if block_index == 1:
            raise RuntimeError("simulated interruption")
        return real_score_block(model, block_index, cached_states, score_config)

    monkeypatch.setattr(sensitivity_cli, "score_block", interrupt_after_block_zero)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sensitivity_cli._run_loaded_score(interrupted_model, config, clean_ids, resumed_dir)
    block_zero_before = (resumed_dir / "block-000.json").read_bytes()

    def reject_block_zero(model, block_index, cached_states, score_config):
        if block_index == 0:
            raise AssertionError("verified block 0 was recomputed")
        return real_score_block(model, block_index, cached_states, score_config)

    monkeypatch.setattr(sensitivity_cli, "score_block", reject_block_zero)
    resumed = sensitivity_cli._run_loaded_score(interrupted_model, config, clean_ids, resumed_dir)

    torch.manual_seed(19)
    uninterrupted_model = _tiny_llada(n_layers=2, d_model=4, vocab_size=11)
    monkeypatch.setattr(sensitivity_cli, "score_block", real_score_block)
    uninterrupted = sensitivity_cli._run_loaded_score(
        uninterrupted_model, config, clean_ids, uninterrupted_dir
    )

    assert (resumed_dir / "block-000.json").read_bytes() == block_zero_before
    assert json.dumps(resumed, sort_keys=True, separators=(",", ":")) == json.dumps(
        uninterrupted, sort_keys=True, separators=(",", ":")
    )
    for name in ("states.json", "block-000.json", "block-001.json", "manifest.json"):
        assert (resumed_dir / name).read_bytes() == (uninterrupted_dir / name).read_bytes()
    manifest = json.loads((resumed_dir / "manifest.json").read_text())
    assert manifest["welford_updates_per_module"] == {"full": 1, "A": 1, "B": 1}
    assert manifest["mask_variant_count"] == 16
    assert all(entry["mask_entry_count"] == 7 * 16 for entry in manifest["completed_blocks"])


# Mutation caught: treating a corrupt completed-block checksum as an incomplete block hides damaged science data.
def test_score_resume_hard_fails_on_corrupt_block_checksum(tmp_path, monkeypatch):
    torch.manual_seed(23)
    model = _tiny_llada(n_layers=2, d_model=4, vocab_size=11)
    config = _toy_score_config(blocks=2, timesteps=(0.5,))
    artifact_dir = tmp_path / "artifacts"
    real_score_block = sensitivity_cli.score_block

    def interrupt_after_block_zero(model, block_index, cached_states, score_config):
        if block_index == 1:
            raise RuntimeError("simulated interruption")
        return real_score_block(model, block_index, cached_states, score_config)

    monkeypatch.setattr(sensitivity_cli, "score_block", interrupt_after_block_zero)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sensitivity_cli._run_loaded_score(model, config, _toy_clean_ids(), artifact_dir)

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_blocks"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(sensitivity_cli, "score_block", real_score_block)

    with pytest.raises(ValueError, match="checksum"):
        sensitivity_cli._run_loaded_score(model, config, _toy_clean_ids(), artifact_dir)


# Mutation caught: hashing only model metadata lets resume mix masks from different dense weights.
def test_score_resume_rejects_changed_model_digest(tmp_path, monkeypatch):
    torch.manual_seed(29)
    model = _tiny_llada(n_layers=2, d_model=4, vocab_size=11)
    config = _toy_score_config(blocks=2, timesteps=(0.5,))
    artifact_dir = tmp_path / "artifacts"
    real_score_block = sensitivity_cli.score_block

    def interrupt_after_block_zero(model, block_index, cached_states, score_config):
        if block_index == 1:
            raise RuntimeError("simulated interruption")
        return real_score_block(model, block_index, cached_states, score_config)

    monkeypatch.setattr(sensitivity_cli, "score_block", interrupt_after_block_zero)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sensitivity_cli._run_loaded_score(model, config, _toy_clean_ids(), artifact_dir)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1)
    monkeypatch.setattr(sensitivity_cli, "score_block", real_score_block)

    with pytest.raises(ValueError, match="model_digest"):
        sensitivity_cli._run_loaded_score(model, config, _toy_clean_ids(), artifact_dir)


@pytest.fixture(scope="module")
def completed_toy_score_artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("completed-score")
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    sensitivity_cli._run_loaded_score(
        model,
        _toy_score_config(blocks=1, timesteps=(0.5,)),
        _toy_clean_ids(),
        artifact_dir,
    )
    return {path.name: path.read_bytes() for path in artifact_dir.iterdir()}


def _restore_toy_score_artifacts(tmp_path, saved):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for name, contents in saved.items():
        (artifact_dir / name).write_bytes(contents)
    return artifact_dir


def _rewrite_completed_block(artifact_dir, masks, metadata):
    block_path = artifact_dir / "block-000.json"
    receipt = save_mask_block(block_path, masks, metadata)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_blocks"][0].update(receipt)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


# Mutation caught: selecting a nonzero lambda applies the old risk mask instead of pure Mean.
def test_apply_mean_masks_preserves_survivors_and_reports_actual_rowwise_sparsity(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    layers = find_layers(model.model.transformer.blocks[0])
    before = {name: layer.weight.detach().clone() for name, layer in layers.items()}
    saved_masks, _ = load_mask_block(artifact_dir / "block-000.json")

    report = sensitivity_cli._apply_mean_masks(model, artifact_dir, 0.6)

    expected_pruned = 0
    expected_total = sum(layer.weight.numel() for layer in layers.values())
    assert report["method"] == "mean_dlm_sensitivity"
    assert report["lambda_zero_equals_mean"] is True
    assert report["requested_sparsity"] == 0.6
    assert report["module_count"] == 7
    assert report["aggregation"] == {
        "sequence_count": 8,
        "timestep_count": 1,
        "updates_per_module": 1,
        "welford_updates_per_module": {"A": 1, "B": 1, "full": 1},
    }
    for module in report["modules"]:
        name = module["module"]
        mask = saved_masks[f"{name}|lambda=0|sparsity=0.6"]
        weight = layers[name].weight.detach()
        expected_pruned += mask.sum().item()
        assert torch.count_nonzero(weight[mask]) == 0
        torch.testing.assert_close(weight[~mask], before[name][~mask])
        assert module["pruned_weight_count"] == mask.sum().item()
        assert module["total_weight_count"] == mask.numel()
        assert module["actual_sparsity"] == pytest.approx(mask.float().mean().item())
        assert module["pruned_per_row_min"] == math.floor(mask.shape[1] * 0.6)
        assert module["pruned_per_row_max"] == math.floor(mask.shape[1] * 0.6)
    assert report["pruned_weight_count"] == expected_pruned
    assert report["total_weight_count"] == expected_total
    assert report["actual_sparsity"] == pytest.approx(expected_pruned / expected_total)
    assert report["sparsity_delta"] == pytest.approx(
        expected_pruned / expected_total - 0.6
    )


def test_apply_mean_masks_rejects_unavailable_sparsity(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    with pytest.raises(ValueError, match="not present"):
        sensitivity_cli._apply_mean_masks(model, artifact_dir, 0.61)


def test_materialize_mean_writes_new_checkpoint_and_machine_readable_report(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    output_dir = tmp_path / "mean-60"

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    report = sensitivity_cli._materialize_loaded_mean(
        model, Tokenizer(), artifact_dir, 0.6, output_dir
    )

    assert report["checkpoint"] == str(output_dir)
    assert json.loads((output_dir / "mean_dlm_pruning.json").read_text()) == report
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "tokenizer.json").is_file()

    with pytest.raises(FileExistsError, match="nonempty"):
        sensitivity_cli._materialize_loaded_mean(
            model, Tokenizer(), artifact_dir, 0.6, output_dir
        )


# Mutation caught: direct materialization computes scores but forgets to apply the
# requested row-wise Mean mask before saving.
def test_materialize_mean_direct_scores_and_prunes_requested_sparsity(tmp_path):
    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    torch.manual_seed(37)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    config = _toy_score_config(blocks=1, timesteps=(0.5,))
    output_dir = tmp_path / "mean-25"

    report = sensitivity_cli._materialize_loaded_mean_direct(
        model, Tokenizer(), config, _toy_clean_ids(), 0.25, output_dir
    )

    layers = find_layers(model.model.transformer.blocks[0])
    assert report["requested_sparsity"] == 0.25
    assert report["source"] == "direct_dlm_gradient_scoring"
    assert report["module_count"] == 7
    assert report["pruned_weight_count"] == sum(
        layer.weight.shape[0] * math.floor(layer.weight.shape[1] * 0.25)
        for layer in layers.values()
    )
    for module in report["modules"]:
        weight = layers[module["module"]].weight.detach()
        expected_per_row = math.floor(weight.shape[1] * 0.25)
        assert torch.count_nonzero(weight == 0, dim=1).unique().tolist() == [
            expected_per_row
        ]
        assert module["pruned_per_row_min"] == expected_per_row
        assert module["pruned_per_row_max"] == expected_per_row
    assert json.loads((output_dir / "mean_dlm_pruning.json").read_text()) == report
    assert (output_dir / "model.safetensors").is_file()


def test_run_materialize_loads_bound_model_and_copies_llada_support(
    tmp_path, completed_toy_score_artifacts, monkeypatch
):
    import main_llada
    import model as model_package
    import transformers

    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    output_dir = tmp_path / "mean-60"
    config = _toy_score_config(blocks=1, timesteps=(0.5,))
    calls = {}

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    def load_model(_, model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        torch.manual_seed(31)
        return _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    def load_tokenizer(_, model_id, **kwargs):
        calls["tokenizer"] = (model_id, kwargs)
        return Tokenizer()

    monkeypatch.setattr(sensitivity_cli, "_validate_production_score_config", lambda _: None)
    monkeypatch.setattr(sensitivity_cli, "_model_cuda_devices", lambda _: (0,))
    monkeypatch.setattr(model_package.LLaDAModelLM, "from_pretrained", classmethod(load_model))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(load_tokenizer))
    monkeypatch.setattr(
        main_llada,
        "copy_llada_support_files",
        lambda source, destination: calls.setdefault("support", (source, destination)),
    )

    report = sensitivity_cli.run_materialize(
        config, artifact_dir, 0.6, output_dir
    )

    assert calls["model"] == (
        "tiny",
        {
            "revision": "test-revision",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        },
    )
    assert calls["tokenizer"] == (
        "tiny",
        {"revision": "test-revision", "trust_remote_code": True},
    )
    assert calls["support"] == ("tiny", str(output_dir))
    assert report["actual_sparsity"] == pytest.approx(
        report["pruned_weight_count"] / report["total_weight_count"]
    )


def test_run_materialize_direct_loads_calibration_and_copies_support(
    tmp_path, monkeypatch
):
    import main_llada
    import model as model_package
    import transformers
    from lib import data

    output_dir = tmp_path / "mean-25"
    config = _toy_score_config(blocks=1, timesteps=(0.5,))
    calls = {}

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    def load_model(_, model_id, **kwargs):
        torch.manual_seed(41)
        return _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    monkeypatch.setattr(sensitivity_cli, "_validate_production_score_config", lambda _: None)
    monkeypatch.setattr(sensitivity_cli, "_model_cuda_devices", lambda _: (0,))
    monkeypatch.setattr(model_package.LLaDAModelLM, "from_pretrained", classmethod(load_model))
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda *_args, **_kwargs: Tokenizer()),
    )
    monkeypatch.setattr(
        data,
        "get_loaders",
        lambda *args, **kwargs: ([(ids, None) for ids in _toy_clean_ids()], None),
    )
    monkeypatch.setattr(
        main_llada,
        "copy_llada_support_files",
        lambda source, destination: calls.setdefault("support", (source, destination)),
    )

    report = sensitivity_cli.run_materialize_direct(config, 0.25, output_dir)

    assert report["requested_sparsity"] == 0.25
    assert report["actual_sparsity"] == pytest.approx(
        report["pruned_weight_count"] / report["total_weight_count"]
    )
    assert calls["support"] == ("tiny", str(output_dir))


def test_score_resume_rejects_self_consistent_module_rename(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    masks, metadata = load_mask_block(artifact_dir / "block-000.json")
    original = next(iter(metadata["module_shapes"]))
    forged = f"forged.{original}"
    metadata["module_shapes"][forged] = metadata["module_shapes"].pop(original)
    metadata["module_diagnostics"][forged] = metadata["module_diagnostics"].pop(original)
    masks = {
        (name.replace(f"{original}|", f"{forged}|", 1) if name.startswith(f"{original}|") else name): mask
        for name, mask in masks.items()
    }
    _rewrite_completed_block(artifact_dir, masks, metadata)
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="module|metadata|audit"):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


def test_score_resume_rejects_nested_numeric_alias(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    masks, metadata = load_mask_block(artifact_dir / "block-000.json")
    module_name = next(iter(metadata["module_shapes"]))
    metadata["module_shapes"][module_name][0] = float(
        metadata["module_shapes"][module_name][0]
    )
    _rewrite_completed_block(artifact_dir, masks, metadata)
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="module|metadata|audit"):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


def test_score_resume_rejects_mutated_state_file_bytes(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    states_path = artifact_dir / "states.json"
    states = json.loads(states_path.read_text())
    states["states"][0]["sequence_index"] = 9
    states_path.write_text(json.dumps(states, sort_keys=True, separators=(",", ":")))
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="state"):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


def test_score_resume_rejects_changed_actual_config_and_digest(
    tmp_path, completed_toy_score_artifacts
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    config = _toy_score_config(blocks=1, timesteps=(0.5,))
    config["reliability"]["minimum_mean_split_mask_jaccard"] = 0.94
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="config_digest"):
        sensitivity_cli._run_loaded_score(
            model, config, _toy_clean_ids(), artifact_dir
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("block_index", 1),
        ("path", "block-999.json"),
        ("sha256", "0" * 64),
        ("byte_length", 1),
        ("module_count", 8),
        ("update_count", 2),
        ("welford_updates", {"full": 2, "A": 1, "B": 1}),
        ("mask_variant_count", 15),
        ("mask_entry_count", 111),
    ],
)
def test_score_resume_rejects_every_corrupt_manifest_entry_audit_field(
    tmp_path, completed_toy_score_artifacts, field, bad_value
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_blocks"][0][field] = bad_value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="block|manifest|audit|checksum|size|path"):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model", {"id": "other", "revision": "test-revision"}),
        ("model_revision", "other"),
        ("model_digest", "0" * 64),
        ("config_digest", "0" * 64),
        ("state_digest", "0" * 64),
        ("state_artifact", {"path": "other.json"}),
        ("block_count", 2),
        ("modules_per_block", 8),
        ("updates_per_module", 2),
        ("welford_updates_per_module", {"full": 2, "A": 1, "B": 1}),
        ("lambdas", [0.0, 1.0]),
        ("sparsities", [0.5]),
        ("score_definition", "corrupt"),
        ("calibration", {"seed": 999}),
        ("mask_variant_count", 15),
    ],
)
def test_score_resume_rejects_every_corrupt_manifest_binding_field(
    tmp_path, completed_toy_score_artifacts, field, bad_value
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match=field):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


@pytest.mark.parametrize(
    "field",
    [
        "model_revision",
        "model_digest",
        "config_digest",
        "state_digest",
        "block_index",
        "module_count",
        "update_count",
        "welford_updates",
        "mask_variant_count",
        "lambdas",
        "sparsities",
        "score_definition",
        "calibration",
        "module_shapes",
        "module_diagnostics",
        "mask_entries",
    ],
)
def test_score_resume_rejects_every_corrupt_block_metadata_field(
    tmp_path, completed_toy_score_artifacts, field
):
    artifact_dir = _restore_toy_score_artifacts(tmp_path, completed_toy_score_artifacts)
    block_path = artifact_dir / "block-000.json"
    masks, metadata = load_mask_block(block_path)
    if field == "module_shapes":
        metadata[field].pop(next(iter(metadata[field])))
    elif field == "module_diagnostics":
        metadata[field].pop(next(iter(metadata[field])))
    elif field == "mask_entries":
        masks.pop(next(iter(masks)))
    elif field == "block_index":
        metadata[field] = 1
    elif field in ("module_count", "update_count", "mask_variant_count"):
        metadata[field] += 1
    elif field == "welford_updates":
        metadata[field] = {"full": 2, "A": 1, "B": 1}
    elif field in ("lambdas", "sparsities"):
        metadata[field] = metadata[field][:-1]
    elif field == "calibration":
        metadata[field]["seed"] += 1
    else:
        metadata[field] = "corrupt"
    save_mask_block(block_path, masks, metadata)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["completed_blocks"][0]
    entry["sha256"] = hashlib.sha256(block_path.read_bytes()).hexdigest()
    entry["byte_length"] = block_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    torch.manual_seed(31)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)

    with pytest.raises(ValueError, match="block|metadata|audit|mask"):
        sensitivity_cli._run_loaded_score(
            model,
            _toy_score_config(blocks=1, timesteps=(0.5,)),
            _toy_clean_ids(),
            artifact_dir,
        )


@pytest.mark.parametrize("fault", ["post_write", "nested_alias", "wrong_mask"])
def test_score_write_validates_before_manifest_or_stat_release(tmp_path, monkeypatch, fault):
    torch.manual_seed(37)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    config = _toy_score_config(blocks=1, timesteps=(0.5,))
    artifact_dir = tmp_path / "artifacts"
    real_score_block = sensitivity_cli.score_block
    real_save = sensitivity_cli.save_mask_block
    real_validate = sensitivity_cli._validate_block_artifact
    statistic = None
    alive_during_validation = []

    def tracked_score_block(*args, **kwargs):
        nonlocal statistic
        result, caches = real_score_block(*args, **kwargs)
        statistic = weakref.ref(next(iter(result["statistics"].values()))["mu"])
        return result, caches

    def corrupt_after_write(path, masks, metadata):
        if fault == "post_write":
            expected = real_save(path, masks, metadata)
            contents = path.read_bytes()
            path.write_bytes(contents[:-1] + (b"{" if contents[-1:] != b"{" else b"}"))
        elif fault == "nested_alias":
            corrupt_metadata = copy.deepcopy(metadata)
            module_name = next(iter(corrupt_metadata["module_diagnostics"]))
            corrupt_metadata["module_diagnostics"][module_name]["masks"][
                "lambda=0|sparsity=0.5"
            ]["mean_disagreement_count"] = False
            expected = real_save(path, masks, corrupt_metadata)
        else:
            corrupt_masks = {name: mask.clone() for name, mask in masks.items()}
            mask = next(iter(corrupt_masks.values()))
            pruned = torch.nonzero(mask[0]).flatten()[0]
            kept = torch.nonzero(~mask[0]).flatten()[0]
            mask[0, pruned] = False
            mask[0, kept] = True
            expected = real_save(path, corrupt_masks, metadata)
        return expected

    def tracked_validate(*args, **kwargs):
        alive_during_validation.append(statistic() is not None)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(sensitivity_cli, "score_block", tracked_score_block)
    monkeypatch.setattr(sensitivity_cli, "save_mask_block", corrupt_after_write)
    monkeypatch.setattr(sensitivity_cli, "_validate_block_artifact", tracked_validate)

    with pytest.raises(ValueError, match="written block|checksum|invalid|metadata"):
        sensitivity_cli._run_loaded_score(model, config, _toy_clean_ids(), artifact_dir)
    assert alive_during_validation == [True]
    assert json.loads((artifact_dir / "manifest.json").read_text())["completed_blocks"] == []


@pytest.mark.parametrize(
    ("keys", "bad_value", "message"),
    [
        (("model", "id"), "other", "model.id"),
        (("stage0", "model_block_count"), True, "stage0.model_block_count"),
        (("calibration", "seed"), False, "calibration.seed"),
        (("calibration", "sequence_indices"), [False, 1, 2, 3, 4, 5, 6, 7], "calibration.sequence_indices"),
        (("calibration", "sequence_count"), True, "calibration.sequence_count"),
        (("calibration", "sequence_length"), 255, "calibration.sequence_length"),
        (("calibration", "timesteps"), [0.05] * 10, "calibration.timesteps"),
        (("calibration", "split_a"), [0, 1, 2, True], "calibration.split_a"),
        (("calibration", "split_b"), [4, 5, 6, True], "calibration.split_b"),
        (("scoring", "gradient_microbatch_size"), True, "scoring.gradient_microbatch_size"),
        (("scoring", "lambdas"), [True, 0.5, 1.0], "scoring.lambdas"),
        (("scoring", "sparsities"), [0.5, 0.6, 0.7, True], "scoring.sparsities"),
        (("scoring", "decision_sparsities"), [0.5, 0.6, True], "scoring.decision_sparsities"),
        (("reliability", "spearman_sample_size"), True, "reliability.spearman_sample_size"),
        (("reliability", "minimum_median_split_sigma_spearman"), True, "reliability.minimum_median_split_sigma_spearman"),
        (("reliability", "minimum_mean_split_mask_jaccard"), True, "reliability.minimum_mean_split_mask_jaccard"),
        (("reliability", "top_sigma_fraction"), True, "reliability.top_sigma_fraction"),
    ],
)
def test_score_production_contract_rejects_changes_before_cuda_or_model_work(
    tmp_path, monkeypatch, keys, bad_value, message
):
    config = json.loads(Path("codex/time_risk_sensitivity/config.json").read_text())
    config[keys[0]][keys[1]] = bad_value
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA checked before score contract")),
    )

    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        sensitivity_cli.run_score(config, tmp_path)


def test_score_production_contract_accepts_committed_pilot():
    config = json.loads(Path("codex/time_risk_sensitivity/config.json").read_text())
    sensitivity_cli._validate_production_score_config(config)


def test_score_production_contract_accepts_16_by_512_calibration_profile():
    config = json.loads(
        Path("codex/calibration_16x512/config.json").read_text()
    )

    sensitivity_cli._validate_production_score_config(config)


def test_repeat_gradient_square_check_real_32_block_tiny_model_covers_all_outcomes(
    monkeypatch,
):
    torch.manual_seed(41)
    model = _tiny_llada(n_layers=32, d_model=4, vocab_size=11)
    config = _toy_score_config(blocks=32, timesteps=(0.5,))
    cached_states = _toy_cached_states(model, config)

    passed = sensitivity_cli._repeat_gradient_square_check(model, config, cached_states)
    assert passed["status"] == "passed"
    assert passed["finite"]
    assert passed["within_tolerance"]

    real_gradients = sensitivity_cli.block_state_gradients

    def checked_with_second_result(transform):
        calls = 0

        def wrapper(*args, **kwargs):
            nonlocal calls
            calls += 1
            gradients, output = real_gradients(*args, **kwargs)
            if calls == 2:
                gradients = {name: value.clone() for name, value in gradients.items()}
                first_name = next(iter(gradients))
                transform(gradients[first_name].view(-1))
            return gradients, output

        return wrapper

    monkeypatch.setattr(
        sensitivity_cli,
        "block_state_gradients",
        checked_with_second_result(lambda values: values.__setitem__(0, values[0] + 1)),
    )
    tolerance_failure = sensitivity_cli._repeat_gradient_square_check(
        model, config, cached_states
    )
    assert tolerance_failure["status"] == "failed"
    assert tolerance_failure["finite"]
    assert not tolerance_failure["within_tolerance"]

    monkeypatch.setattr(
        sensitivity_cli,
        "block_state_gradients",
        checked_with_second_result(lambda values: values.__setitem__(0, float("inf"))),
    )
    nonfinite_failure = sensitivity_cli._repeat_gradient_square_check(
        model, config, cached_states
    )
    assert nonfinite_failure["status"] == "failed"
    assert not nonfinite_failure["finite"]
    assert not nonfinite_failure["within_tolerance"]


# Mutation caught: pooling weights or sparsities can pass a gate whose predeclared module summaries fail.
def test_score_block_stage1_gate_uses_unweighted_module_thresholds_and_signal_stop():
    config = _toy_score_config()
    diagnostics = [
        {
            "rho_sigma_A_sigma_B": {"value": rho, "reason": None},
            "masks": {
                f"lambda=1|sparsity={sparsity:g}": {"split_jaccard": overlap}
                for sparsity, overlap in zip((0.5, 0.6, 0.7), overlaps)
            },
        }
        for rho, overlaps in (
            (0.4, (0.94, 0.95, 0.96)),
            (0.5, (0.96, 0.95, 0.94)),
            (0.9, (0.95, 0.95, 0.95)),
        )
    ]

    gate = sensitivity_cli._stage1_gate(diagnostics, config, all_useful_masks_identical=False)
    stopped = sensitivity_cli._stage1_gate(diagnostics, config, all_useful_masks_identical=True)
    undefined_diagnostics = copy.deepcopy(diagnostics)
    undefined_diagnostics[0]["rho_sigma_A_sigma_B"] = {
        "value": None,
        "reason": "constant vector",
    }
    undefined = sensitivity_cli._stage1_gate(
        undefined_diagnostics, config, all_useful_masks_identical=False
    )

    assert gate["median_module_split_sigma_spearman"] == pytest.approx(0.5)
    assert gate["mean_split_mask_jaccard"] == {"0.5": 0.95, "0.6": 0.95, "0.7": 0.95}
    assert gate["reliability_passed"]
    assert gate["passed"]
    assert not stopped["passed"]
    assert stopped["stop_reason"] == "all useful Time-Risk masks are identical to Mean"
    assert not undefined["reliability_passed"]
    assert undefined["undefined_split_sigma_spearman_modules"] == 1
    assert undefined["stop_reason"] == "split-half reliability gate failed"
