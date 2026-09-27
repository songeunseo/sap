import math

from experiments.dlm_role_validation.core import allocate_grid
from experiments.dlm_role_wanda25_full.run import GRID, TARGET_INDEX


def test_target25_grid_and_exact_budget():
    shapes = [(4, 16), (8, 32), (4, 48)]
    curves = [[float(level * (module + 1)) for level in range(6)] for module in range(3)]
    allocation = allocate_grid(curves, shapes, GRID, TARGET_INDEX)
    expected = sum(rows * int(cols * .25) for rows, cols in shapes)
    assert allocation["pruned"] == expected
    assert allocation["budget_error"] == 0
    assert math.isclose(GRID[TARGET_INDEX], .25)
