def test_get_llm_honors_requested_sequence_length(monkeypatch):
    import main_llada

    class Model:
        class Config:
            max_sequence_length = 4096

        config = Config()

    monkeypatch.setattr(
        main_llada.LLaDAModelLM,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: Model()),
    )

    assert main_llada.get_llm("test-model", seqlen=256).seqlen == 256


def test_get_llm_honors_requested_revision(monkeypatch):
    import main_llada

    captured = {}

    class Model:
        class Config:
            max_sequence_length = 4096

        config = Config()

    def from_pretrained(cls, *args, **kwargs):
        captured.update(kwargs)
        return Model()

    monkeypatch.setattr(
        main_llada.LLaDAModelLM,
        "from_pretrained",
        classmethod(from_pretrained),
    )

    main_llada.get_llm("test-model", revision="frozen-revision")

    assert captured["revision"] == "frozen-revision"
