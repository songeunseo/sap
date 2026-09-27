from pathlib import Path

from experiments.dlm_baselines50_gsm8k_mini100 import run as r
from experiments.dlm_multiscale_ac50.artifacts import read


def test_fixed_arm_family():
    assert len(r.ARMS) == 11
    assert r.REFERENCE == "Uniform-row"
    assert len(set(r.ARMS)) == len(r.ARMS)


def test_frozen_requests_are_ids_0_99():
    q = read(r.REQUEST_SOURCE)
    assert [x["example_id"] for x in q["development"]] == list(range(100))
    assert q["protocol_hash"] == read(r.LEGACY_CONFIG)["protocol_hash"]


def test_all_source_manifests_have_common_module_order():
    orders=[]
    for arm,(path,_) in r.CANDIDATES.items():
        m=read(path); r._metadata(arm,m); orders.append([x["name"] for x in m["entries"]])
    assert all(x == orders[0] for x in orders)


def test_budget_contract():
    for arm,(path,engine) in r.CANDIDATES.items():
        meta=r._metadata(arm,read(path))
        if engine == "wanda": assert meta["actual_pruned"] == r.TARGET
        else: assert meta["actual_pruned"] == 3_489_662_724


def test_fixed_contrast_family():
    contrasts=[(r.REFERENCE,a) for a in r.ARMS if a != r.REFERENCE]
    assert len(contrasts)==10 and all(a != b for a,b in contrasts)


def test_source_files_exist():
    assert r.REQUEST_SOURCE.is_file() and r.LEGACY_CONFIG.is_file() and r.MASK_SOURCE.is_file()
    assert all(Path(p).is_file() for p,_ in r.CANDIDATES.values())


def test_explicit_llada_mask_id_is_frozen():
    assert read(r.MASK_SOURCE)["mask_id"] == 126336
