import pytest
import torch
import yaml

from experiments.dlm_loss_aggregation.core import (
    EffectAccumulator,
    analyze_scores,
    mask_sha256,
    pack_mask,
    pairwise_diagnostics,
    profile_pairwise_spearman,
    rowwise_mask,
    unpack_mask,
)
from experiments.dlm_loss_aggregation.run import (
    BlockEffectCollector,
    build_calibration_manifest,
    evaluate_in_memory_sequence,
    historical_state_digest,
    load_config,
    require_historical_digest,
    validate_config,
    validate_smoke_module,
    weighted_pair_summaries,
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


def test_full_analysis_returns_the_exact_masks_used_for_xor():
    scores = {
        "sum": torch.tensor([[-4.0, 1.0, -2.0, 3.0]]),
        "abs": torch.tensor([[4.0, 1.0, 2.0, 3.0]]),
        "square": torch.tensor([[16.0, 1.0, 4.0, 9.0]]),
    }

    diagnostics, masks, distributions = analyze_scores(scores)

    assert torch.equal(masks["sum"], torch.tensor([[True, False, True, False]]))
    assert diagnostics["pairs"][0]["mask_xor"] == 0.5
    assert torch.allclose(distributions["sign_consistency"], torch.ones(1, 4))
    assert torch.allclose(distributions["spike_ratio"], torch.ones(1, 4))


def test_pair_summaries_are_weighted_by_matrix_size():
    rows = [
        {"module_type": "q", "left": "sum", "right": "abs", "num_weights": 1, "spearman": 1.0, "mask_xor": 0.0},
        {"module_type": "q", "left": "sum", "right": "abs", "num_weights": 3, "spearman": 0.5, "mask_xor": 0.5},
        {"module_type": "ff", "left": "sum", "right": "abs", "num_weights": 4, "spearman": 0.0, "mask_xor": 1.0},
    ]

    summaries = weighted_pair_summaries(rows)

    by_scope = {(row["scope"], row["module_type"]): row for row in summaries}
    assert by_scope[("module_type", "q")]["spearman"] == pytest.approx(0.625)
    assert by_scope[("module_type", "q")]["mask_xor"] == pytest.approx(0.375)
    assert by_scope[("weighted_overall", "all")]["spearman"] == pytest.approx(0.3125)
    assert by_scope[("weighted_overall", "all")]["mask_xor"] == pytest.approx(0.6875)


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


def test_spearman_profiler_reports_runtime_memory_and_all_pairs():
    profile = profile_pairwise_spearman(
        {
            "sum": torch.tensor([[3.0, 1.0, 2.0]]),
            "abs": torch.tensor([[1.0, 2.0, 3.0]]),
            "square": torch.tensor([[1.0, 4.0, 9.0]]),
        }
    )

    assert profile["elapsed_seconds"] >= 0
    assert profile["peak_rss_kib"] > 0
    assert profile["rss_delta_kib"] >= 0
    assert len(profile["pairs"]) == 3


def test_calibration_manifest_preserves_historical_state_semantics_and_digest():
    config = {
        "model": {"id": "model", "revision": "revision"},
        "dataset": {"id": "dataset", "configuration": "raw", "split": "train"},
        "calibration": {
            "seed": 0,
            "sequence_indices": [7],
            "sequence_length": 4,
            "epsilon": 0.001,
            "timesteps": [0.5, 0.75],
        },
    }

    manifest = build_calibration_manifest(
        [torch.tensor([[10, 11, 12, 13]])], mask_id=99, config=config
    )

    assert manifest["global_random_seed"] == 0
    assert manifest["states"][0] == {
        "timestep_index": 0,
        "timestep": 0.5,
        "sequence_index": 7,
        "mask_seed": 0,
        "p_mask": 0.5005,
        "clean_ids": [[10, 11, 12, 13]],
        "noisy_ids": [[99, 11, 99, 99]],
        "mask": [[True, False, True, True]],
    }
    assert historical_state_digest(manifest) == (
        "145036d65a4c92169d8a6e0b580ce0c446d26cf890ad1651280581523538bbe5"
    )


def test_calibration_digest_mismatch_stops_before_scoring():
    with pytest.raises(ValueError, match="digest mismatch"):
        require_historical_digest({"states": []}, "wrong")


def test_block_collector_uses_one_backward_clears_gradients_and_preserves_weights():
    layers = {
        "a": torch.nn.Linear(2, 2, bias=False),
        "b": torch.nn.Linear(2, 2, bias=False),
    }
    with torch.no_grad():
        layers["a"].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        layers["b"].weight.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    dense = {name: layer.weight.detach().clone() for name, layer in layers.items()}
    value = torch.tensor([[1.0, 2.0]])

    collector = BlockEffectCollector(layers)
    with collector.hooks():
        (layers["a"](value) + layers["b"](value)).sum().backward()
        collector.step()
    scores = collector.finalize()

    assert torch.equal(scores["a"]["sum"], torch.tensor([[-1.0, 0.0], [0.0, -2.0]]))
    assert torch.equal(scores["b"]["abs"], torch.tensor([[2.0, 6.0], [4.0, 10.0]]))
    assert torch.equal(scores["b"]["square"], torch.tensor([[4.0, 36.0], [16.0, 100.0]]))
    assert all(layer.weight.grad is None for layer in layers.values())
    assert all(torch.equal(layers[name].weight, dense[name]) for name in layers)


def test_block_collector_rejects_a_backward_missing_one_module():
    layers = {
        "used": torch.nn.Linear(2, 2, bias=False),
        "missing": torch.nn.Linear(2, 2, bias=False),
    }
    collector = BlockEffectCollector(layers)

    with collector.hooks():
        layers["used"](torch.ones(1, 2)).sum().backward()
        with pytest.raises(RuntimeError, match="missing gradients"):
            collector.step()


def test_real_smoke_allows_one_sign_sum_and_identical_rankings():
    dense = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    scores = {
        "sum": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "abs": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "square": torch.tensor([[1.0, 4.0, 9.0, 16.0]]),
    }

    result = validate_smoke_module(
        dense_before=dense,
        dense_after=dense.clone(),
        loss=torch.tensor(1.25),
        scores=scores,
        count=1,
        expected_count=1,
    )

    assert result["negative_sum_fraction"] == 0.0
    abs_square = next(
        pair for pair in result["pairs"] if pair["left"] == "abs"
    )
    assert abs_square["spearman"] == 1.0
    assert abs_square["mask_xor"] == 0.0


def test_production_config_freezes_experiment_1a(tmp_path):
    source = load_config("experiments/dlm_loss_aggregation/config.yaml")
    validate_config(source)
    source["model"]["revision"] = "different"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(ValueError, match="model.revision"):
        validate_config(load_config(path))


def test_evaluation_sequence_reloads_dense_and_releases_dlm_masks():
    config = load_config("experiments/dlm_loss_aggregation/config.yaml")
    loaded = []
    pruned = []
    masks = {
        "dlm_sum": {"hash": "sum", "payload": b"sum"},
        "dlm_abs": {"hash": "abs", "payload": b"abs"},
        "dlm_square": {"hash": "square", "payload": b"square"},
    }

    def load_dense():
        model = object()
        loaded.append(model)
        return model

    def prune(model, method, mask):
        pruned.append((model, method, None if mask is None else mask["hash"]))
        return "baseline-hash" if mask is None else mask["hash"]

    def evaluate(model, method):
        return {"accuracy": 0.25, "num_examples": 4, "eval_seconds": 1.0}

    results = evaluate_in_memory_sequence(config, masks, load_dense, prune, evaluate)

    assert [row["method"] for row in results] == config["evaluation"]["order"]
    assert len({id(model) for model in loaded}) == 6
    assert [method for _, method, _ in pruned] == config["evaluation"]["order"][1:]
    assert masks == {}
    assert results[1]["mask_hash"] == "sum"
    assert results[-1]["calibration_label"].startswith("standard reference")
