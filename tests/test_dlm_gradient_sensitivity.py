import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import dlm_gradient_sensitivity as sensitivity_cli

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
def test_score_block_matches_literal_direct_gradient_loop_and_masks():
    torch.manual_seed(11)
    model = _tiny_llada(n_layers=1, d_model=4, vocab_size=11)
    config = _toy_score_config()
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

    result, next_states = score_block(model, 0, states, config)

    assert result["update_count"] == 2
    assert list(next_states) == list(states)
    for name, statistics in expected.items():
        for statistic in ("mu", "sigma", "sigma_A", "sigma_B"):
            torch.testing.assert_close(result["statistics"][name][statistic], statistics[statistic])
        for risk_lambda in (0.0, 0.25, 0.5, 1.0):
            score = statistics["mu"] + risk_lambda * statistics["sigma"]
            for sparsity in (0.5, 0.6, 0.7, 0.75):
                assert torch.equal(
                    result["masks"][(name, risk_lambda, sparsity)],
                    rowwise_prune_mask(score, sparsity),
                )
                split_a_mask = rowwise_prune_mask(
                    statistics["mu_A"] + risk_lambda * statistics["sigma_A"], sparsity
                )
                split_b_mask = rowwise_prune_mask(
                    statistics["mu_B"] + risk_lambda * statistics["sigma_B"], sparsity
                )
                diagnostic = result["diagnostics"][name]["masks"][
                    f"lambda={risk_lambda:g}|sparsity={sparsity:g}"
                ]
                assert diagnostic["split_A_sha256"] == pack_mask(split_a_mask)["sha256"]
                assert diagnostic["split_B_sha256"] == pack_mask(split_b_mask)["sha256"]


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

    assert gate["median_module_split_sigma_spearman"] == pytest.approx(0.5)
    assert gate["mean_split_mask_jaccard"] == {"0.5": 0.95, "0.6": 0.95, "0.7": 0.95}
    assert gate["reliability_passed"]
    assert gate["passed"]
    assert not stopped["passed"]
    assert stopped["stop_reason"] == "all useful Time-Risk masks are identical to Mean"
