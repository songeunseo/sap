import numpy as np
import torch

from experiments.dlm_super_outlier_statistics.core import (
    correlation,
    partial_rank_correlation,
    read_contribution,
    residual_addition,
    tensor_statistics,
)


def test_tensor_statistics_separates_selected_channel():
    x = torch.ones(2, 4)
    x[:, 2] = 4
    result = tensor_statistics(x, channel=2)
    assert result["dominant_channel"] == 2
    assert np.isclose(result["super_share"], 16 / 19)
    assert np.isclose(result["excluded_dominant_share"], 1 / 3)


def test_read_contribution_is_exact_decomposition():
    x = torch.tensor([[2.0, 3.0]])
    weight = torch.tensor([[5.0, 7.0], [11.0, 13.0]])
    bias = torch.tensor([1.0, -2.0])
    y = x @ weight.T + bias
    result = read_contribution(x, y, weight, bias, channel=0)
    assert result["decomposition_error"] < 1e-10
    assert result["bias_present"]


def test_residual_addition_is_exact_decomposition():
    before = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]])
    branch = torch.tensor([[[3.0, 5.0], [7.0, 11.0]]])
    result = residual_addition(before, before + branch, channel=1)
    assert result["decomposition_error"] < 1e-10


def test_partial_rank_correlation_removes_group_confounding():
    layers = np.repeat(np.arange(4), 4)
    types = np.tile(np.array(["a", "b", "c", "d"]), 4)
    x = layers.astype(float) + (types == "d") * 0.1
    y = x.copy()
    raw = correlation(x, y)["spearman"]
    partial = partial_rank_correlation(x, y, layers, types)["spearman"]
    assert raw == 1.0
    assert partial is None or abs(partial) < 0.3
