"""Read frozen experiment artifacts; write only this paper workspace.

No model loading, experiment mutation, retries, or external publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = REPO / "experiments/dlm_crosschain_control50/output"
OLD = REPO / "experiments/dlm_multiscale_ac50/output"
MINI = REPO / "experiments/dlm_multi_validation10050/output"
FIRST = RUN.parent / "output_attempt1_scheduler_failure_20260927"
ARMS = ("A", "Multi", "Cross", "CrossMatched")
FAMILY = ("Multi-A", "Multi-Cross", "Multi-CrossMatched")
SOURCES = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def checked(path, expected):
    actual = sha(path)
    require(actual == expected, f"Changed source: {path}")
    SOURCES[str(path)] = actual


def write(name, value):
    path = HERE / name
    require(path.parent == HERE, "Write outside paper workspace")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(text)
    temp.replace(path)


def statistics(reference, candidate, bootstrap=False):
    require(len(reference) == len(candidate) and len(reference) > 0, "Unpaired samples")
    gain = sum(b and not a for a, b in zip(reference, candidate))
    loss = sum(a and not b for a, b in zip(reference, candidate))
    discordant = gain + loss
    # Independent exact binomial sum, rather than the experiment's scipy call.
    p = min(1.0, (2 * sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1))) / (2 ** discordant)) if discordant else 1.0
    out = dict(gain=gain, loss=loss, net=gain-loss,
               difference_pp=100*(gain-loss)/len(reference), exact_mcnemar_p=p)
    if bootstrap:
        import numpy as np
        delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
        rng = np.random.default_rng(20260927)
        samples = []
        for _ in range(20):
            indices = rng.integers(0, len(delta), size=(500, len(delta)))
            samples.extend(delta[indices].mean(axis=1))
        out["paired_bootstrap_95pp_unadjusted"] = (100*np.quantile(samples, [.025, .975])).tolist()
    return out


def adjust(family):
    previous = 0.0
    for i, name in enumerate(sorted(family, key=lambda k: family[k]["exact_mcnemar_p"])):
        previous = max(previous, min(1.0, (len(family)-i)*family[name]["exact_mcnemar_p"]))
        family[name]["holm_p"] = previous


def strict_paths(folder, expected_ids):
    found = {}
    for path in folder.glob("*.json"):
        require(re.fullmatch(r"\d{4}\.json", path.name) is not None, f"Noncanonical checkpoint: {path}")
        i = int(path.stem)
        require(path.name == f"{i:04d}.json" and i not in found, f"Duplicate ID: {path}")
        found[i] = path
    require(set(found) == set(expected_ids), f"Missing/unexpected checkpoints in {folder}")
    return [found[i] for i in expected_ids]


def exact_history(previous, current):
    keys = ("example_id", "doc_hash", "prompt_hash", "target_hash", "reference_answer",
            "evaluation_config_hash", "generated_text", "extracted_answer", "correct")
    for old in previous:
        new = current[old["example_id"]]
        require(all(old[k] == new[k] for k in keys), f"Historical reproduction changed: {old['example_id']}")


def task():
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol
    config = read(RUN / "config.json")
    return task_and_protocol(read(config["legacy_config"]))[0]


def verify_folder(folder, requests, fingerprint, evaluator):
    from experiments.dlm_multiscale_ac50.evaluation import grade, validate_prediction
    paths = strict_paths(folder / "examples", [r["example_id"] for r in requests])
    rows = []
    for request, path in zip(requests, paths):
        saved = read(path)
        row = saved["row"]
        require(saved["fingerprint"] == fingerprint and saved["row_sha256"] == digest(row), f"Checkpoint hash/identity: {path}")
        validate_prediction(row, request, fingerprint["protocol_hash"])
        result = grade(evaluator, request, row["generated_text"])
        require(all(row[k] == result[k] for k in ("correct", "extracted_answer")), f"Official regrade differs: {path}")
        rows.append(row)
    result = read(folder / "results.json")
    checked(folder / "predictions.json", result["predictions_sha256"])
    require(result["status"] == "complete" and result["fingerprint"] == fingerprint,
            f"Result identity: {folder}")
    require(read(folder / "predictions.json") == rows, f"Aggregate/checkpoint mismatch: {folder}")
    require(result["total"] == len(rows) and result["correct"] == sum(r["correct"] for r in rows), f"Score mismatch: {folder}")
    return rows


def verify_mini(evaluator):
    report = read(MINI / "report.json")
    require(report["status"] == read(MINI / "execution.json")["status"] == "complete", "Mini incomplete")
    requests = read(MINI / "requests.json")["development"]
    require([r["example_id"] for r in requests] == report["document_ids"] and len(set(report["document_ids"])) == 100, "Mini IDs")
    rows = {}
    for arm in ("Multi", "A", "Uniform"):
        folder = MINI / "gsm8k/validation100" / arm
        fp = read(folder / "results.json")["fingerprint"]
        require(fp["config_sha256"] == sha(MINI / "config.json") and fp["requests_sha256"] == sha(MINI / "requests.json"), "Mini fingerprint")
        rows[arm] = verify_folder(folder, requests, fp, evaluator)
        require(report["scores"][arm] == {"correct": sum(r["correct"] for r in rows[arm]), "total": 100}, "Mini totals")
    pairs = {}
    for a, b in (("A", "Multi"), ("Uniform", "Multi"), ("Uniform", "A")):
        pairs[f"{b}_minus_{a}"] = statistics([r["correct"] for r in rows[a]], [r["correct"] for r in rows[b]])
    adjust(pairs)
    for key, value in pairs.items():
        old = report["paired"][key]
        require(value["gain"] == old["gained"] and value["loss"] == old["lost"], "Mini discordances")
        require(all(abs(value[k]-old[k]) < 1e-12 for k in ("net", "exact_mcnemar_p", "holm_p")), "Mini paired statistics")
    development = {}
    old_requests = read(OLD / "requests.json")["development"]
    for arm in ("A", "Multi"):
        folder = OLD / "gsm8k/development" / arm
        result = read(folder / "results.json")
        verified = verify_folder(folder, old_requests, result["fingerprint"], evaluator)
        development[arm] = {"correct": sum(r["correct"] for r in verified), "total": len(verified)}
    uniform = read(MINI / "cached_development.json")["uniform"]
    from experiments.dlm_multiscale_ac50.evaluation import grade, validate_prediction
    require([r["example_id"] for r in uniform] == [r["example_id"] for r in old_requests], "Uniform development IDs")
    protocol_hash = read(RUN / "config.json")["protocol_hash"]
    for row, req in zip(uniform, old_requests):
        validate_prediction(row, req, protocol_hash)
        graded = grade(evaluator, req, row["generated_text"])
        require(all(row[k] == graded[k] for k in ("correct", "extracted_answer")), "Uniform regrade")
    development["Uniform"] = {"correct": sum(r["correct"] for r in uniform), "total": len(uniform)}
    return {"scores": report["scores"], "development": development, "paired": pairs, "source": str(MINI / "report.json")}


def freeze_inputs():
    paths = [RUN / n for n in ("config.json", "allocation.json", "requests.json", "cpu_validation.json", "audit_pass.json")]
    paths += [RUN / "candidates" / a / n for a in ARMS for n in ("model_identity.json", "mask_manifest.json")]
    paths += [MINI / "report.json", MINI / "config.json", MINI / "requests.json", MINI / "cached_development.json"]
    require(FIRST.is_dir(), "Preserved first attempt missing")
    paths += sorted((FIRST / "attempts").glob("*.json")) + sorted((FIRST / "costs").glob("*.json"))
    paths += [OLD / "gsm8k/development" / a / "predictions.json" for a in ("A", "Multi")]
    paths += [MINI / "gsm8k/validation100" / a / "predictions.json" for a in ("A", "Multi", "Uniform")]
    paths += sorted(RUN.parent.glob("*.py")) + sorted(RUN.parent.glob("*.sh"))
    paths += [HERE / n for n in ("closeout.py", "paper.template.md", "contract.json", "claim-map.md")]
    snapshot = {str(p): sha(p) for p in paths}
    if (HERE / "inputs.json").exists():
        require(read(HERE / "inputs.json") == snapshot, "Frozen writing inputs changed; review before replacing snapshot")
    else:
        write("inputs.json", snapshot)
    return snapshot


def verify_frozen():
    for path, expected in read(HERE / "inputs.json").items():
        checked(path, expected)
    cpu, audit = read(RUN / "cpu_validation.json"), read(RUN / "audit_pass.json")
    require(cpu["status"] == audit["status"] == "passed", "Source audit failed")
    require(cpu["config_sha256"] == audit["config_sha256"] == sha(RUN / "config.json"), "Source audit config")
    require(cpu["allocation_sha256"] == sha(RUN / "allocation.json"), "Source allocation hash")
    require(audit["cpu_validation_sha256"] == sha(RUN / "cpu_validation.json"), "Source audit receipt stale")
    require(cpu["code_sources"] == audit["code_sources"], "Source audited code mismatch")
    for path, expected in cpu["code_sources"].items():
        checked(path, expected)


def verify_shards(evaluator, complete):
    config = read(RUN / "config.json")
    requests = read(RUN / "requests.json")["development"]
    require([r["example_id"] for r in requests] == list(range(1319)), "Full requests")
    all_rows = {a: [] for a in ARMS}
    for arm in ARMS:
        identity = read(RUN / "candidates" / arm / "model_identity.json")
        for s in range(11):
            receipt = RUN / "done" / f"{arm}_{s:02d}.json"
            if not complete and not receipt.exists():
                continue
            folder = RUN / "gsm8k" / arm / f"shard{s:02d}"
            checked(folder / "results.json", read(receipt)["result_sha256"])
            fp = dict(config_sha256=sha(RUN / "config.json"), requests_sha256=sha(RUN / "requests.json"),
                      mask_identity=identity["mask_identity"], sparse_model_sha256=identity["sparse_model_sha256"],
                      split="full", shard=s, protocol_hash=config["protocol_hash"])
            all_rows[arm].extend(verify_folder(folder, requests[s*128:min((s+1)*128, 1319)], fp, evaluator))
        if complete:
            expected_paths = {RUN / "gsm8k" / arm / f"shard{i//128:02d}" / "examples" / f"{i:04d}.json" for i in range(1319)}
            require(set((RUN / "gsm8k" / arm).glob("shard*/examples/*.json")) == expected_paths, f"Unexpected shard/checkpoint: {arm}")
            require([r["example_id"] for r in all_rows[arm]] == list(range(1319)), f"Full count: {arm}")
            require(read(RUN / "gsm8k" / arm / "predictions.json") == all_rows[arm], "Full aggregate mismatch")
    return all_rows


def cost_accounting(roots):
    output = {}
    for label, root in roots.items():
        attempts = {p.stem: read(p) for p in (root / "attempts").glob("*.json")}
        costs = {p.stem: read(p) for p in (root / "costs").glob("*.json")}
        for name in attempts.keys() & costs.keys():
            require(attempts[name]["job"] == costs[name]["job"], "Cost/attempt job mismatch")
        output[label] = {"attempt_count": len(attempts), "cost_count": len(costs),
            "recorded_forwards": sum(r["forward_calls"] for r in costs.values()),
            "sum_worker_wall_seconds": sum(r["wall_seconds"] for r in costs.values()),
            "missing_costs": sorted(attempts.keys() - costs.keys()),
            "orphan_costs": sorted(costs.keys() - attempts.keys()),
            "statuses": dict(Counter(r["status"] for r in costs.values()))}
    gaps = any(r["missing_costs"] or r["orphan_costs"] for r in output.values())
    return {"runs": output, "recorded_forwards_total": sum(r["recorded_forwards"] for r in output.values()),
            "complete_accounting": not gaps, "counts_are_lower_bounds": gaps,
            "scope": "Current full evaluation plus preserved scheduler-failure attempt; excludes earlier reused probe/bank/model construction"}


def verify_full(evaluator):
    state, report = read(RUN / "execution.json"), read(RUN / "report.json")
    require(state["status"] == report["status"] == "complete" and not state["workers"], "Full run is not complete")
    config, allocation = read(RUN / "config.json"), read(RUN / "allocation.json")
    require(report["config_sha256"] == sha(RUN / "config.json") and report["allocation_sha256"] == sha(RUN / "allocation.json"), "Report source identity")
    require(report["primary_family"] == list(FAMILY), "Primary family changed")
    expected_jobs = {"teacher", *[f"init_{a}" for a in ARMS], *[f"{a}_{s:02d}" for a in ARMS for s in range(11)]}
    require(set(state["completed"]) == expected_jobs, "Completion job coverage")
    for arm in ARMS:
        manifest = read(RUN / "candidates" / arm / "mask_manifest.json")
        ident = read(RUN / "candidates" / arm / "model_identity.json")
        require(manifest["pruned"] == 3489660928 and manifest["config_sha256"] == sha(RUN / "config.json"), "Mask budget/config")
        require(manifest["allocation_sha256"] == sha(RUN / "allocation.json"), "Mask allocation")
        require([e["selected_mask"]["prune_per_row"] for e in manifest["entries"]] == allocation["allocations"][arm]["row_counts"], "Row counts")
        require(digest([[e["name"], e["shape"], e["selected_mask"]["mask_sha256"]] for e in manifest["entries"]]) == ident["mask_identity"], "Mask identity")
        for entry in manifest["entries"]:
            checked(entry["selected_mask"]["path"], entry["selected_mask"]["file_sha256"])
        init = read(RUN / "done" / f"init_{arm}.json")
        require(init["mask_identity"] == ident["mask_identity"] and init["model_sha256"] == ident["sparse_model_sha256"], "Init identity")
        for split, expected in init["readout_sha256"].items():
            checked(RUN / "readouts" / arm / f"{split}.json", expected)
    teacher = read(RUN / "done/teacher.json")
    checked(RUN / "readouts/dense/fresh.json", teacher["readout_sha256"])
    rows = verify_shards(evaluator, complete=True)
    exposed = config["crosschain_control"]["exposed_ids"]
    primary = config["crosschain_control"]["primary_ids"]
    require(len(set(exposed)) == len(exposed) == 200 and len(set(primary)) == len(primary) == 1119, "Sample size")
    require(set(primary).isdisjoint(exposed) and set(primary) | set(exposed) == set(range(1319)), "Sample partition")
    for arm in ("A", "Multi"):
        history = read(OLD / "gsm8k/development" / arm / "predictions.json") + read(MINI / "gsm8k/validation100" / arm / "predictions.json")
        require(len(history) == 200 and {r["example_id"] for r in history} == set(exposed), "Historical IDs")
        exact_history(history, {r["example_id"]: r for r in rows[arm]})
    scores, comparisons, categories = {}, {}, {}
    for group, ids in {"full_1319": list(range(1319)), "previously_seen_200": exposed, "primary_remaining_1119": primary}.items():
        scores[group] = {a: {"correct": sum(rows[a][i]["correct"] for i in ids), "total": len(ids)} for a in ARMS}
        require(scores[group] == report["scores"][group], f"Scores differ: {group}")
        comparisons[group] = {f"Multi-{a}": statistics([rows[a][i]["correct"] for i in ids], [rows["Multi"][i]["correct"] for i in ids], bootstrap=True) for a in ("A", "Cross", "CrossMatched")}
        if group == "primary_remaining_1119":
            adjust(comparisons[group])
        for name, values in comparisons[group].items():
            saved = report["comparisons"][group][name]
            for key, value in values.items():
                if isinstance(value, list):
                    require(all(abs(x-y) < 1e-10 for x,y in zip(value, saved[key])), f"CI differs: {group}/{name}")
                else:
                    require(abs(value-saved[key]) < 1e-10, f"Statistic differs: {group}/{name}/{key}")
        categories[group] = {a: dict(Counter("correct" if rows[a][i]["correct"] else "strict_invalid" if rows[a][i]["extracted_answer"] in ("[invalid]", "", None) else "valid_wrong" for i in ids)) for a in ARMS}
    from experiments.dlm_crosschain_control50.core import response_metrics
    fresh = config["banks"]["fresh"]
    checked(fresh["path"], fresh["sha256"])
    dense = read(RUN / "readouts/dense/fresh.json")["values"]
    diagnostics = {}
    for arm in ARMS:
        value = response_metrics(read(RUN / "readouts" / arm / "fresh.json")["values"], dense, read(fresh["path"]))
        require(value == report["diagnostics"]["fresh"][arm], f"Fresh diagnostics differ: {arm}")
        diagnostics[arm] = value["mean"]
    costs = cost_accounting({"current": RUN, "preserved_first_attempt": FIRST})
    return {"scores": scores, "comparisons": comparisons, "error_categories": categories,
            "fresh_diagnostics": diagnostics, "costs": costs, "beta": allocation["beta"],
            "historical_reproduction": {"A": 200, "Multi": 200, "generated_text_and_grading_identical": True},
            "wall_hours_current_run": (state["ended"]-state["started"])/3600,
            "source_report": str(RUN / "report.json"), "primary_ids": primary,
            "limitations": report["limitations"]}


def mini_table(mini):
    lines = ["| Method | Development100 correct | Separate100 correct |", "|---|---:|---:|"]
    lines += [f"| {a} | {mini['development'][a]['correct']}/100 | {mini['scores'][a]['correct']}/100 |" for a in ("Multi", "A", "Uniform")]
    lines += ["", "| Contrast | Gain/loss | Net | Exact p | Holm p |", "|---|---:|---:|---:|---:|"]
    for name, r in mini["paired"].items():
        lines.append(f"| {name.replace('_minus_', '−')} | {r['gain']}/{r['loss']} | {r['net']:+d} | {r['exact_mcnemar_p']:.5g} | {r['holm_p']:.5g} |")
    return "\n".join(lines)


def render(mini, full=None):
    pending = "**[미측정 / PENDING: full-run completion and verification required.]**"
    replacements = dict(RUN_STATUS="Preparation copy; full results are pending.", ABSTRACT_RESULT="**[PENDING: the full comparison is still running.]**",
                        MINI_TABLE=mini_table(mini), BETA=f"{read(RUN / 'allocation.json')['beta']:.12g}",
                        **{k: pending for k in ("MAIN_TABLE", "MAIN_STATS", "RESULT_READING", "DIAGNOSTICS", "COSTS")})
    if full is not None:
        replacements["RUN_STATUS"] = "Paper v0; full results passed the documented CPU verification. Scientific review remains open."
        replacements["ABSTRACT_RESULT"] = "The verified effect estimates and paired uncertainty are reported in Section 5; their interpretation is limited to this frozen experiment."
        table = ["| Arm | Full1319 | Previously examined200 | Primary1119 |", "|---|---:|---:|---:|"]
        for a in ARMS:
            table.append(f"| {a} | " + " | ".join(f"{full['scores'][g][a]['correct']}/{full['scores'][g][a]['total']}" for g in ("full_1319", "previously_seen_200", "primary_remaining_1119")) + " |")
        replacements["MAIN_TABLE"] = "\n".join(table)
        stats = ["| Primary contrast | Gain/loss | Difference (pp) | Exact p | Holm p | Unadjusted 95% CI (pp) |", "|---|---:|---:|---:|---:|---:|"]
        descriptions = []
        for name, r in full["comparisons"]["primary_remaining_1119"].items():
            lo, hi = r["paired_bootstrap_95pp_unadjusted"]
            stats.append(f"| {name} | {r['gain']}/{r['loss']} | {r['difference_pp']:+.3f} | {r['exact_mcnemar_p']:.5g} | {r['holm_p']:.5g} | [{lo:.3f}, {hi:.3f}] |")
            verdict = "shows a positive difference" if r["net"] > 0 else "shows a negative difference" if r["net"] < 0 else "has zero net difference"
            evidence = "passes the Holm-adjusted 0.05 threshold" if r["holm_p"] < .05 else "does not pass the Holm-adjusted 0.05 threshold"
            descriptions.append(f"{name} {verdict} ({r['net']:+d} answers) and {evidence}.")
        replacements["MAIN_STATS"] = "\n".join(stats)
        replacements["RESULT_READING"] = " ".join(descriptions) + " These pairwise findings do not establish equivalence, universal necessity, or generalization. The contribution claim must be reviewed jointly with the control results and the uncertainty intervals."
        diag = ["| Arm | A | Natural C | Cross C | Query CE | Response sign flip rate |", "|---|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            diag.append(f"| {a} | " + " | ".join(f"{full['fresh_diagnostics'][a][k]:.6g}" for k in ("A", "C_natural", "C_cross", "query_CE", "response_sign_flip_rate")) + " |")
        replacements["DIAGNOSTICS"] = "\n".join(diag)
        costs = full["costs"]
        qualifier = "a lower bound because some attempt records lack matching costs" if costs["counts_are_lower_bounds"] else "complete for the recorded current and preserved first attempts"
        replacements["COSTS"] = f"The current run took {full['wall_hours_current_run']:.3f} wall-clock hours. Current plus preserved first-attempt records contain {costs['recorded_forwards_total']:,} model forward calls. This count is {qualifier}. It excludes earlier construction of reused probes, calibration artifacts and masks, so it is not the end-to-end method cost."
    text = (HERE / "paper.template.md").read_text()
    for key, value in replacements.items():
        token = "{{" + key + "}}"
        require(token in text, f"Missing template token: {key}")
        text = text.replace(token, value)
    require(not re.search(r"\{\{[A-Z_]+\}\}", text), "Unfilled template token")
    return text


def finalize():
    require(read(RUN / "execution.json")["status"] == "complete", "Refusing finalization before completion")
    verify_frozen()
    evaluator = task()
    mini = verify_mini(evaluator)
    full = verify_full(evaluator)
    verify_frozen()
    text = render(mini, full)
    require("[미측정" not in text and "[PENDING:" not in text, "Unfilled measured-results slot")
    if (HERE / "paper-v0.md").exists():
        require((HERE / "paper-v0.md").read_text() == text, "Existing paper-v0 differs; preserve edits and review before regeneration")
    evidence = {"status": "verified", "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mini": mini, "full": full, "sources": dict(SOURCES),
                "table_sources": {"mini": "mini/scores and mini/paired", "full": "full/scores", "primary": "full/comparisons/primary_remaining_1119", "diagnostics": "full/fresh_diagnostics", "costs": "full/costs"}}
    write("closeout.json", evidence)
    write("closeout.md", "# Full experiment closeout\n\nCPU verification passed: canonical checkpoint names, exact coverage, fingerprints, official regrading, paired statistics, mask file hashes, fresh diagnostics and historical A/Multi200 reproduction.\n\n" + render(mini, full).split("## 5. Results", 1)[1].split("## 6. Discussion", 1)[0] + "\n## Error categories and cost gaps\n\n```json\n" + json.dumps({"errors": full["error_categories"], "costs": full["costs"]}, ensure_ascii=False, indent=2) + "\n```\n\nScientific interpretation and Obsidian synchronization remain pending agent review.\n")
    write("paper-v0.md", text)
    write("state.json", {"status": "v0_generated_needs_review", "paper_sha256": sha(HERE / "paper-v0.md"),
                         "closeout_sha256": sha(HERE / "closeout.json"), "finished": time.time(),
                         "next": "Review interpretation and novelty, sync completed results to Obsidian, select a follow-up proposal; no new experiment is authorized here."})
    print("Verified full results and generated paper-v0.md", flush=True)


def process_matches(state):
    try:
        stat = Path(f"/proc/{state['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and stat[19] == state["start_ticks"]
    except (FileNotFoundError, KeyError):
        return False


def watch():
    require(bool(os.environ.get("TMUX")), "Watch must run inside tmux")
    with (HERE / "watch.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen()
        write("state.json", {"status": "waiting_for_full_run", "pid": os.getpid(), "started": time.time(), "source": str(RUN)})
        print("Waiting for existing full controller; no model work will be launched.", flush=True)
        deadline = time.monotonic() + 24*3600
        while time.monotonic() < deadline:
            state = read(RUN / "execution.json")
            if state["status"] == "complete":
                finalize()
                return
            require(state["status"] == "running", f"Experiment stopped: {state.get('status')}; {state.get('error')}")
            if not process_matches(state):
                # Controller can exit between the first state read and /proc inspection.
                state = read(RUN / "execution.json")
                if state["status"] == "complete":
                    continue
                raise RuntimeError("Experiment controller disappeared before completion")
            time.sleep(60)
        raise RuntimeError("Completion wait exceeded 24h; experiment left unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "watch", "finalize"))
    args = parser.parse_args()
    try:
        if args.command == "check":
            freeze_inputs()
            verify_frozen()
            evaluator = task()
            mini = verify_mini(evaluator)
            rows = verify_shards(evaluator, complete=False)
            write("paper-preparation.md", render(mini))
            receipt = {"status": "passed", "full_status": read(RUN / "execution.json")["status"],
                       "mini": mini, "verified_completed_shard_questions": {a: len(r) for a,r in rows.items()},
                       "sources": dict(SOURCES), "time": time.time()}
            write("preflight.json", receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k not in ("sources", "mini")}))
        elif args.command == "watch":
            watch()
        else:
            finalize()
    except Exception as exc:
        if args.command != "check":
            write("state.json", {"status": "needs_attention", "error": str(exc), "time": time.time(), "experiment_unchanged": True})
        raise


if __name__ == "__main__":
    main()
