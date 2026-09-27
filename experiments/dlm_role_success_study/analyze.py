"""Joint, paired outcome study; no model runs or selection of new allocations."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from experiments.dlm_role_bundle_mini100 import run as source

ROOT = Path(__file__).resolve().parent
SEED = 20260913
BOOTSTRAPS = 20000


def groups(a, r):
    a, r = np.asarray(a, dtype=bool), np.asarray(r, dtype=bool)
    return dict(role_only=(~a & r), aggregate_only=(a & ~r), both_correct=(a & r), both_wrong=(~a & ~r))


def decompose(a, r, add, revert):
    a, r = np.asarray(a, dtype=int), np.asarray(r, dtype=int)
    add, revert = np.asarray(add, dtype=int), np.asarray(revert, dtype=int)
    g = groups(a, r)
    result = {}
    for k in range(add.shape[1]):
        by_group = {}
        for name, mask in g.items():
            by_group[name] = dict(n=int(mask.sum()), add_correct=int(add[mask, k].sum()),
                revert_correct=int(revert[mask, k].sum()),
                add_benefit=int((add[:, k]-a)[mask].sum()),
                role_background_benefit=int((r-revert[:, k])[mask].sum()))
        result[str(k)] = dict(by_group=by_group, add_benefit=int((add[:, k]-a).sum()),
                             role_background_benefit=int((r-revert[:, k]).sum()))
    return result


def contrasts(a, r, add, revert):
    gain = r-a
    forward = add-a[:, None]
    reverse = r[:, None]-revert
    return np.column_stack((gain, gain-forward.sum(1), reverse.sum(1)-gain, (reverse-forward).mean(1)))


def bootstrap(values):
    # The complete per-question vector is resampled together, never bundles independently.
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(BOOTSTRAPS//500):
        ids = rng.integers(0, len(values), size=(500, len(values)))
        estimates.append(values[ids].mean(axis=1))
    samples = np.concatenate(estimates)
    return np.quantile(samples, [.025, .975], axis=0).T.tolist()


def design_rank():
    eye = np.eye(6, dtype=int)
    x = np.vstack((np.zeros(6, dtype=int), np.ones(6, dtype=int), eye, 1-eye))
    additive = np.column_stack((np.ones(len(x)), x))
    pairs = list(itertools.combinations(range(6), 2))
    pairwise = np.column_stack((additive, np.column_stack([x[:, i]*x[:, j] for i, j in pairs])))
    return dict(observed_coalitions=len(x), possible_coalitions=2**6,
        additive_columns=additive.shape[1], additive_rank=int(np.linalg.matrix_rank(additive)),
        pairwise_columns=pairwise.shape[1], pairwise_rank=int(np.linalg.matrix_rank(pairwise)),
        pairwise_nullity=int(pairwise.shape[1]-np.linalg.matrix_rank(pairwise)),
        interpretation="Rank deficiency prevents unique pairwise attribution even with infinitely many questions at these same design rows.")


def hypothetical_power():
    rng = np.random.default_rng(SEED)
    results = []
    for n in (100, 1319):
        for discordance in (.10, .20):
            for effect in (.01, .02, .03, .05):
                draws = rng.multinomial(n, [(discordance-effect)/2, 1-discordance, (discordance+effect)/2], size=20000)
                d = draws[:, 0]+draws[:, 2]
                p = np.minimum(1, 2*binom.cdf(np.minimum(draws[:, 0], draws[:, 2]), d, .5))
                rates = {}
                for label, alpha in (("alpha05", .05), ("bonferroni12", .05/12)):
                    rate = float(np.mean(p <= alpha))
                    rates[label] = dict(power=rate, monte_carlo_se=float(np.sqrt(rate*(1-rate)/len(p))))
                results.append(dict(n=n, hypothetical_discordance=discordance, hypothetical_effect=effect, **rates))
    return results


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "config.json").exists():
        raise RuntimeError("Completed/partial study exists; preserve original artifacts")
    c = source.validate()
    baselines = {k: source.rows(v) for k, v in c["baselines"].items()}
    candidates = {m: source.completed(m, c) for m in source.METHODS}
    if any(x is None for x in candidates.values()):
        raise RuntimeError("all 12 completed outcomes required")
    for data in baselines.values():
        source.validate_rows(data, baselines["aggregate"], c["protocol_hash"])
    paths = [source.ROOT / "config.json", source.ROOT / "results.json", source.ROOT / "maintenance/path_fix_20260913/amendment.json",
             source.ROOT / "run.py", source.ROOT / "core.py", Path(__file__), ROOT / "test_analyze.py"]
    paths += [Path(p) for p in c["baselines"].values()]
    paths += [source.ROOT / m / f for m in source.METHODS for f in ("predictions.jsonl", "results.json")]
    sources = {str(p): source.sha256(p) for p in paths}
    source.write(ROOT / "config.json", dict(status="frozen_before_joint_analysis", seed=SEED, bootstrap=BOOTSTRAPS,
        source_sha256=sources, purpose="exploratory mechanism study, no allocation selection", resampling_unit="question with all14 outcomes jointly",
        power="fixed hypothetical effects1/2/3/5pp,discordance10/20%,n100/1319,20k simulations; not observed power; Bonferroni not Holm power"), frozen=True)
    a = np.array([x["correct"] for x in baselines["aggregate"]], dtype=int)
    r = np.array([x["correct"] for x in baselines["role"]], dtype=int)
    add = np.column_stack([np.array([x["correct"] for x in candidates[f"add_b{k}"]], dtype=int) for k in range(6)])
    revert = np.column_stack([np.array([x["correct"] for x in candidates[f"revert_b{k}"]], dtype=int) for k in range(6)])
    g = groups(a, r)
    partitions = decompose(a, r, add, revert)
    values = contrasts(a, r, add, revert)
    labels = ["endpoint_role_minus_aggregate", "endpoint_gain_minus_sum_add_benefits", "sum_reverse_benefits_minus_endpoint_gain", "mean_background_benefit_difference"]
    cis = bootstrap(values)
    global_stats = {name: dict(mean=float(values[:, i].mean()), sum=float(values[:, i].sum()),
        bootstrap_ci95_unadjusted=cis[i], interpretation="exploratory; conditional on frozen generation and masks") for i, name in enumerate(labels)}
    per_question = []
    for i in range(len(a)):
        per_question.append(dict(example_id=i, group=next(name for name, mask in g.items() if mask[i]),
            aggregate_correct=int(a[i]), role_correct=int(r[i]), add_correct=add[i].tolist(), revert_correct=revert[i].tolist(),
            add_correct_count=int(add[i].sum()), revert_correct_count=int(revert[i].sum()),
            observed_correct_methods=int(a[i]+r[i]+add[i].sum()+revert[i].sum()),
            doc_hash=baselines["aggregate"][i]["doc_hash"]))
    outcomes = np.column_stack((a, r, add, revert))
    descriptive = dict(group_counts={name: int(mask.sum()) for name, mask in g.items()},
        always_correct=int(np.all(outcomes==1, axis=1).sum()), always_wrong=int(np.all(outcomes==0, axis=1).sum()),
        outcome_varies_across14=int(np.any(outcomes != outcomes[:, :1], axis=1).sum()),
        role_only_ids=np.flatnonzero(g["role_only"]).tolist(), aggregate_only_ids=np.flatnonzero(g["aggregate_only"]).tolist(),
        role_only_correct_in_no_add=int(np.sum(g["role_only"] & (add.sum(1)==0))),
        role_only_lost_in_any_revert=int(np.sum(g["role_only"] & (revert.sum(1)<6))),
        both_correct_lost_in_any_hybrid=int(np.sum(g["both_correct"] & (outcomes.min(1)==0))),
        both_wrong_rescued_in_any_hybrid=int(np.sum(g["both_wrong"] & (outcomes.max(1)==1))))
    result = dict(status="complete", descriptives=descriptive, bundle_decomposition=partitions,
        global_joint_contrasts=global_stats, design_identifiability=design_rank(), hypothetical_power=hypothetical_power())
    source.write(ROOT / "results.json", result, frozen=True)
    source.write(ROOT / "question_outcomes.json", per_question, frozen=True)
    (ROOT / "report.md").write_text(report(result, per_question))
    for p, h in sources.items():
        if source.sha256(p) != h: raise RuntimeError("input mutated during analysis")
    source.write(ROOT / "receipt.json", dict(status="complete", inputs_unchanged=True,
        outputs={p: source.sha256(ROOT / p) for p in ("config.json", "results.json", "question_outcomes.json", "report.md")}), frozen=True)
    print((ROOT / "report.md").read_text())


def report(result, per_question):
    d = result["descriptives"]
    lines = ["# 기존 Role 성공의 문항별 재분석", "", "## Objective / Hypothesis", "",
        "Role이 새로 맞힌 문항의 회복/소실과 공통정답 손상을 분리한다. 특정bundle이 성공원인이라는 가정 없이 joint outcome과 설계 식별성을 분석한다.", "",
        "## Setup", "", "기존14모델×동일100문항. 새GPU/생성/할당변경없음. 개별총점은 이미 관측한 상태의 탐색분석이며 endpoint로선택한집단은 설명용이다. Source/receipt/identity/strictEM 검사 및입력불변확인.", "",
        "## Result: 실제로 어떤 문제를 잃었나", "", f"기준집단: {d['group_counts']}", "",
        "아래 두 lost 열은 Role에서해당묶음을되돌렸을때 소실된정답수다. 총점순변화와달리 다른문항에서얻은정답은아직차감하지않은gross값이다.", "",
        "| Bundle | add로 Role-only8 중 회복 | revert로 Role-only8 중 소실 | revert로 공통정답16 중 소실 | revert로 기존 Role오답 중 회복 | Role−revert 순효과 |",
        "|---|---:|---:|---:|---:|---:|"]
    for b, p in result["bundle_decomposition"].items():
        g = p["by_group"]
        lines.append(f"| B{b} | {g['role_only']['add_correct']} | {g['role_only']['n']-g['role_only']['revert_correct']} | {g['both_correct']['n']-g['both_correct']['revert_correct']} | {g['aggregate_only']['revert_correct']+g['both_wrong']['revert_correct']} | {p['role_background_benefit']:+d} |")
    lines += ["", f"14개설정모두정답 {d['always_correct']}, 모두오답 {d['always_wrong']}, 정오가바뀐문항 {d['outcome_varies_across14']}.",
        f"Role-only8중단일add어느것에서도회복안된문항 {d['role_only_correct_in_no_add']}; 하나이상revert에서소실된문항 {d['role_only_lost_in_any_revert']}.",
        f"공통정답중hybrid에서하나라도틀린문항 {d['both_correct_lost_in_any_hybrid']}; 공통오답중hybrid에서하나라도맞힌문항 {d['both_wrong_rescued_in_any_hybrid']}.", "",
        "| Role-only 문항ID | 6개add 중 정답수 | 6개revert 중 정답수 |", "|---|---:|---:|"]
    for p in per_question:
        if p["group"] == "role_only": lines.append(f"| {p['example_id']} | {p['add_correct_count']} | {p['revert_correct_count']} |")
    lines += ["", "## Result: 전체변경 비가산성", "", "100문항의14개결과를함께bootstrap했다. CI는개별/탐색용,model/calibration/seed불확실성은포함하지않는다.", ""]
    for name, x in result["global_joint_contrasts"].items():
        lines.append(f"- {name}: {100*x['mean']:+.3f}pp, CI {[round(100*z,3) for z in x['bootstrap_ci95_unadjusted']]}")
    lines += ["", "accuracy는이산결과라additive한연속latent score를threshold해도비가산성이생길수있다. 비가산성은직접적인neural-path상호작용입증이아니다. 여러bundle에서같은문항을잃으면효과합에는중복계산되므로합을전체gain의고유기여분해로쓰지않는다.", "",
        "## Result: 현재설계에서 식별할 수 있는 것", "", str(result["design_identifiability"]), "",
        "같은14설정의문항만늘리면평균효과CI는줄일수있지만rankdeficiency는해결되지않는다. 특정pair상호작용/Shapley기여는추가가정이나새coalition측정없이식별불가.", "",
        "## Result: 검출력 설계민감도", "", "가정한참효과/discordance에대한별도simulation이다. 실제효과추정이나실패원인판정이아니다. Bonferroni수준은보수적인12비교기준이지실제Holm검출력이아니다.", "",
        "| n | 가정discordance | 가정참차이 pp | 단일검정 검출률 | 보수적12비교 검출률 |", "|---|---:|---:|---:|---:|"]
    for p in result["hypothetical_power"]:
        lines.append(f"| {p['n']} | {100*p['hypothetical_discordance']:.0f}% | {100*p['hypothetical_effect']:.0f} | {100*p['alpha05']['power']:.1f}% | {100*p['bonferroni12']['power']:.1f}% |")
    lines += ["", "## Interpretation / Decision", "", "이번분석은기존Role성공의증거를꾸미거나새할당을고르는작업이아니다. Role-only문항과공통문항손상을구분하여다음원인검증대상을정확히한다. 역할분리유지. neural기전은아직미확정.", "",
        "## Next Experiment", "", "단순히같은교환을더많이생성하는것과실제role경로를검증하는것을구분한다. 후자는동일task-state에서masked/unmasked projection perturbation을분리하고정답readout변화를측정해야한다. 현재생성trace/공통입력state가없으므로이보고서에서는수행하지않았다. 기준모델두개와sham/Both재현부터시작하고,semantic-role vs cardinality-matched randomcontrol 및연속taskreadout를포함하는좁은설계가필요하다. 기존WT2 KL negative결과를새로발견한것처럼반복하지않는다.", ""]
    return "\n".join(lines)


if __name__ == "__main__": main()
