import json

from experiments.dream_dense_gsm8k.run import extract_records, load_config


def test_frozen_config_is_matched_not_official_protocol():
    config = load_config()
    assert config["evaluation"]["num_fewshot"] == 5
    assert config["provenance"]["official_base_gsm8k_fewshot"] == 8
    assert config["evaluation"]["max_new_tokens"] == config["evaluation"]["diffusion_steps"] == 256


def test_extract_records_keeps_strict_match_and_sorts():
    samples = [
        {
            "filter": "flexible-extract",
            "doc_id": 0,
            "doc_hash": "d0",
            "prompt_hash": "p0",
            "target_hash": "t0",
            "target": "1",
            "resps": [["one"]],
            "filtered_resps": [["1"]],
            "exact_match": 1,
        },
        {
            "filter": "strict-match",
            "doc_id": 1,
            "doc_hash": "d1",
            "prompt_hash": "p1",
            "target_hash": "t1",
            "target": "2",
            "resps": [["two"]],
            "filtered_resps": [["2"]],
            "exact_match": 1,
        },
        {
            "filter": "strict-match",
            "doc_id": 0,
            "doc_hash": "d0",
            "prompt_hash": "p0",
            "target_hash": "t0",
            "target": "1",
            "resps": [["bad"]],
            "filtered_resps": [["bad"]],
            "exact_match": 0,
        },
    ]
    rows = extract_records(samples, "hash")
    assert [row["example_id"] for row in rows] == [0, 1]
    assert [row["correct"] for row in rows] == [False, True]
    assert json.loads(json.dumps(rows))[0]["protocol_hash"] == "hash"
