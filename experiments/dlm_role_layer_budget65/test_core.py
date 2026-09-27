import numpy as np

from experiments.dlm_role_layer_budget65.core import (
    allocate_by_layer,
    allocate_exact_target,
    layer_indices,
)


def test_exact_target_and_cheapest_increment():
    shapes = [(2, 20), (2, 20)]
    curves = np.array([[0, 1, 2, 3, 4, 5], [0, 10, 20, 30, 40, 50]], float)
    # Each 5pp increment removes two weights. From 50% base, add four weights.
    out = allocate_exact_target(curves, shapes, 44)
    assert out["pruned"] == 44
    assert out["levels"] == [2, 0]


def test_layer_indices_repository_order_independent():
    names = [f"block_{layer:02d}.p{projection}" for layer in range(32) for projection in range(7)]
    groups = layer_indices(names)
    assert len(groups) == 32
    assert groups[31] == list(range(217, 224))


def test_allocate_by_layer_preserves_targets():
    names = [f"block_{layer:02d}.p{projection}" for layer in range(32) for projection in range(7)]
    shapes = [(1, 20)] * len(names)
    curves = np.tile(np.arange(6, dtype=float), (len(names), 1))
    targets = {layer: 91 for layer in range(32)}  # 7 * 10 base + 21 increments.
    out = allocate_by_layer(curves, shapes, names, targets)
    assert out["pruned"] == sum(targets.values())
    assert all(row["pruned"] == 91 for row in out["layers"])
