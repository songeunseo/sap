import numpy as np
import torch

from experiments.dlm_role_causal_decomposition.core import (
    CONDITIONS, logit_additivity_metrics, normalized_abs_contrast,
    role_intervention_outputs,
)


def test_role_outputs_and_energy_matching():
    dense = torch.zeros(len(CONDITIONS), 4, 2)
    sparse = dense + torch.tensor([[[3., 4.], [0., 2.], [6., 8.], [0., 1.]]]).expand_as(dense)
    actual = torch.tensor([True, True, False, False])
    random = {101: torch.tensor([True, False, True, False]),
              202: torch.tensor([False, True, False, True]),
              303: torch.tensor([True, False, False, True])}
    result, audit = role_intervention_outputs(dense, sparse, actual, random)
    assert torch.equal(result[3], sparse[3])
    assert torch.equal(result[4], sparse[4])
    assert torch.equal(result[1, ~actual], dense[1, ~actual])
    assert torch.equal(result[2, actual], dense[2, actual])
    assert np.isclose(result[5, actual].float().norm(), result[6, ~actual].float().norm())
    assert min(audit["masked_scale"], audit["unmasked_scale"]) <= 1
    assert max(audit["masked_scale"], audit["unmasked_scale"]) == 1


def test_logit_additivity_is_zero_for_additive_cells():
    logits = torch.zeros(len(CONDITIONS), 3, 5)
    logits[1] = 1
    logits[2] = 2
    logits[3] = 3
    result = logit_additivity_metrics(logits, torch.tensor([True, False, True]))
    assert result["additivity_residual_norm"] == 0


def test_normalized_abs_contrast_handles_joint_zero_without_epsilon():
    result = normalized_abs_contrast([0, 3, 2], [0, 1, 2])
    assert np.allclose(result, [0, .5, 0])
