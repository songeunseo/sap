import pytest

from experiments.dlm_dual_role_mini100.core import (
    mini_decision,
    same_selected_masks,
)


def _manifest(mask_hash="a"):
    return {"entries": [{"name": "block_00.q_proj",
                           "selected_mask": {"mask_sha256": mask_hash, "pruned": 10}}]}


def test_selected_mask_identity_uses_payload_hash_and_count():
    assert same_selected_masks(_manifest(), _manifest())
    assert not same_selected_masks(_manifest("a"), _manifest("b"))


def test_mini_decision_requires_strict_role_improvement_over_aggregate():
    assert mini_decision(19, 20, 12)["decision"] == "KEEP FOR FULL GSM8K"
    assert mini_decision(20, 20, 12)["decision"] == "STOP"
    assert mini_decision(21, 20, 12)["decision"] == "STOP"
