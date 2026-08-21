from types import SimpleNamespace

import eval_llada
from datasets import Dataset


def test_loglikelihood_tokenizes_without_dataset_serialization(monkeypatch):
    harness = object.__new__(eval_llada.LLaDAEvalHarness)
    harness.tokenizer = lambda text: {"input_ids": [ord(char) for char in text]}
    harness.get_loglikelihood = lambda prefix, target: -float(len(prefix) + len(target))
    harness.suffix_greedy_prediction = lambda prefix, target: True
    monkeypatch.setattr(
        Dataset,
        "from_list",
        lambda _: (_ for _ in ()).throw(AssertionError("Dataset must not serialize the model")),
    )

    result = harness.loglikelihood([SimpleNamespace(args=("ab", " c"))])

    assert result == [(-4.0, 1.0)]
