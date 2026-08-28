from types import SimpleNamespace

import eval_llada
import torch
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


def test_harness_accepts_an_in_memory_model_and_tokenizer(monkeypatch):
    class PlacedModel(torch.nn.Module):
        def to(self, *args, **kwargs):
            raise AssertionError("injected model must not be moved")

    model = PlacedModel()
    tokenizer = object()
    monkeypatch.setattr(
        eval_llada.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("injected model must not be loaded")
        ),
    )
    monkeypatch.setattr(
        eval_llada.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("injected tokenizer must not be loaded")
        ),
    )

    harness = eval_llada.LLaDAEvalHarness(
        model=model,
        tokenizer=tokenizer,
        mc_num=1,
        batch_size=1,
    )

    assert harness.model is model
    assert harness.tokenizer is tokenizer
    assert not model.training
