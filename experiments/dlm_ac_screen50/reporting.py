"""Auditable report construction for the AC mini100 screen.

The builder is deliberately independent of the GPU worker and of the command
line controller.  ``build_report`` accepts already validated prediction rows,
then reads only receipts and metric artifacts needed to describe provenance,
diagnostics, and costs.  It returns ``(report_dict, markdown_text)``; callers
are responsible for writing those values atomically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = (
    "MS-A", "Short", "Path", "All", "Multi",
    "Square-A", "Square-AC", "Vector-A", "Vector-AC", "Exchange-AC",
)
REFERENCES = ("legacy_uniform", "legacy_A", "legacy_AC")
ALL_LABELS = ARMS + REFERENCES
CONTRASTS = (
    ("Multi", "Short"),
    ("Multi", "MS-A"),
    ("Multi", "Path"),
    ("Multi", "All"),
    ("Square-AC", "Square-A"),
    ("Vector-AC", "Vector-A"),
    ("Exchange-AC", "legacy_AC"),
)

_MULTI_KEYS = ("A", "C1", "C2", "C4", "C_path", "C_all", "Multi")
_SQUARE_KEYS = (
    "A", "C", "AC", "common2", "main1_2", "main2_2", "interaction2", "mixed2",
)
_VECTOR_KEYS = ("A", "C", "AC")
_CATEGORIES = ("calibration", "teacher_setup", "probes", "search", "diagnostics", "generation", "other")


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return default


def _json_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _unwrap_rows(value: Any) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Accept a plain row list or an injected ``{rows, ...metadata}`` fixture."""
    if isinstance(value, Mapping):
        raw = value.get("rows", value.get("predictions", []))
        meta = {str(k): v for k, v in value.items() if k not in {"rows", "predictions"}}
    else:
        raw, meta = value, {}
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return [], meta
    return [x for x in raw if isinstance(x, Mapping)], meta


def _expected_ids(meta: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[int]:
    ids = meta.get("expected_ids")
    if not isinstance(ids, list):
        ids = manifest.get("development_ids")
    if not isinstance(ids, list):
        ids = list(range(100))
    return [int(x) for x in ids if isinstance(x, int) and not isinstance(x, bool)]


def _valid_rows(rows: Iterable[Mapping[str, Any]], expected: set[int]) -> tuple[list[Mapping[str, Any]], list[Any]]:
    valid: list[Mapping[str, Any]] = []
    invalid: list[Any] = []
    seen: set[int] = set()
    for row in rows:
        eid = row.get("example_id")
        reason = None
        if not isinstance(eid, int) or isinstance(eid, bool) or eid not in expected:
            reason = "unexpected_or_invalid_example_id"
        elif eid in seen:
            reason = "duplicate_example_id"
        elif type(row.get("correct")) is not bool:
            reason = "missing_boolean_correctness"
        if reason is not None:
            invalid.append({"example_id": eid, "reason": reason})
            continue
        seen.add(eid)
        valid.append(row)
    return valid, invalid


def _candidate_name(label: str) -> str:
    return {"MS-A": "A", "legacy_uniform": "uniform", "legacy_A": "A", "legacy_AC": "AC"}.get(label, label)


def _arm_family(label: str, definition: Mapping[str, Any]) -> str:
    family = definition.get("family")
    if isinstance(family, str):
        return family
    if label.startswith("Square"):
        return "Square"
    if label.startswith("Vector"):
        return "Vector"
    if label.startswith("Exchange"):
        return "Exchange"
    return "Multi"


def _definition_for(label: str, armdefs: Mapping[str, Any], references: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize manifest reference spellings to the runtime display labels."""
    direct = armdefs.get(label)
    if isinstance(direct, Mapping):
        return dict(direct)
    aliases = {
        "legacy_uniform": ("legacy_uniform", "legacy_Uniform", "legacy-Uniform", "uniform"),
        "legacy_A": ("legacy_A", "legacy-a", "A"),
        "legacy_AC": ("legacy_AC", "legacy-ac", "AC"),
    }.get(label, (label,))
    for key in aliases:
        value = references.get(key)
        if isinstance(value, Mapping):
            out = dict(value)
            out.setdefault("provenance", "historical_cached_reference")
            out.setdefault("allocator", "historical reference")
            return out
    return {}


def _path_text(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _identity_from_files(root: Path, multi: Path, label: str) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
    """Find result/model identity, including imported historical references."""
    name = _candidate_name(label)
    if label in REFERENCES:
        cfg = _read(multi / "config.json", {})
        manifests = cfg.get("legacy_manifests", {}) if isinstance(cfg, Mapping) else {}
        entry = manifests.get(name, {}) if isinstance(manifests, Mapping) else {}
        if isinstance(entry, Mapping) and entry.get("identity"):
            fp = {"mask_identity": entry.get("identity"), "sparse_model_sha256": entry.get("sparse_model_sha256")}
            receipt = Path(entry["path"]) if entry.get("path") else None
            return {"fingerprint": fp}, receipt, dict(entry)
        return {}, None, {}
    old = label in {"MS-A", "Short", "Path", "All", "Multi"}
    result_dir = (multi / "gsm8k" / "development" / name) if old else (root / "gsm8k" / name)
    identity_path = result_dir / "identity.json"
    identity = _read(identity_path)
    if not isinstance(identity, Mapping):
        identity = None
        identity_path = None
    candidates_root = multi / "candidates" if old else root / "candidates"
    model = _read(candidates_root / name / "model_identity.json")
    if not isinstance(model, Mapping):
        model = None
    return dict(identity or {}), identity_path, dict(model or {})


def _fingerprint(meta: Mapping[str, Any], identity: Mapping[str, Any] | None, model: Mapping[str, Any] | None) -> dict[str, Any] | None:
    candidate = meta.get("fingerprint")
    if isinstance(candidate, Mapping) and candidate:
        fp = dict(candidate)
    elif isinstance(identity, Mapping) and isinstance(identity.get("fingerprint"), Mapping):
        fp = dict(identity["fingerprint"])
    else:
        fp = {}
    if isinstance(model, Mapping):
        for key in ("mask_identity", "sparse_model_sha256"):
            if key not in fp and model.get(key) is not None:
                fp[key] = model[key]
    return fp if fp else None


def _identity_valid(fp: Mapping[str, Any] | None) -> bool:
    # A mask identity is the minimum physical identity.  Requests/protocol and
    # model hashes are included when the runtime has them, but synthetic CPU
    # fixtures may intentionally provide only the mask identity.
    return isinstance(fp, Mapping) and bool(fp.get("mask_identity"))


def _bank_metadata(root: Path, multi: Path, label: str, definition: Mapping[str, Any]) -> dict[str, Any]:
    explicit = definition.get("bank") or definition.get("bank_identity")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if label in REFERENCES:
        source = definition.get("source")
        return {"id": str(definition.get("label") or label), "path": source}
    family = _arm_family(label, definition)
    if family == "Square":
        path = root / "banks" / "Square_calibration.json"
        return {"id": "Square calibration 40-quartet/160-state bank", "path": _path_text(path)}
    if family == "Vector":
        path = root / "banks" / "Vector_calibration.json"
        if not path.exists():
            path = multi.parent.parent / "dlm_context_response50" / "pairs.json"
        return {"id": "legacy ordered 80-pair bank", "path": _path_text(path)}
    if family == "Exchange":
        path = multi.parent.parent / "dlm_context_response50" / "pairs.json"
        return {"id": "legacy scalar ordered 80-pair bank", "path": _path_text(path)}
    path = multi / "bank_calibration.json"
    return {"id": "legacy Multi 128-state monotone bank", "path": _path_text(path)}


def _source_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    hashes = manifest.get("source_hashes", {})
    return {
        "ledger": manifest.get("source_ledger"),
        "ledger_sha256": manifest.get("source_ledger_sha256"),
        "hashes": dict(hashes) if isinstance(hashes, Mapping) else {},
    }


def _question_links(root: Path, multi: Path, label: str, valid_rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    name = _candidate_name(label)
    if label in REFERENCES:
        source = multi / "cached_development.json"
        return {str(row["example_id"]): str(source) for row in valid_rows if isinstance(row.get("example_id"), int)}
    old = label in {"MS-A", "Short", "Path", "All", "Multi"}
    folder = (multi / "gsm8k" / "development" / name) if old else (root / "gsm8k" / name)
    result: dict[str, str] = {}
    for row in valid_rows:
        eid = row.get("example_id")
        if isinstance(eid, int):
            path = folder / "examples" / f"{eid:04d}.json"
            result[str(eid)] = str(path)
    return result

def _mean_metrics(records: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> dict[str, float | None]:
    values: dict[str, list[float]] = {key: [] for key in keys}
    for record in records:
        mean = record.get("mean", record.get("metrics", record))
        if not isinstance(mean, Mapping):
            continue
        for key in values:
            value = mean.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[key].append(float(value))
    return {key: (sum(v) / len(v) if v else None) for key, v in values.items()}


def _metric_records(folder: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not folder.exists():
        return records
    for path in sorted(folder.rglob("*.json")):
        row = _read(path)
        if isinstance(row, Mapping):
            records.append({"path": str(path), "data": dict(row)})
    return records


def _family_diagnostics(folder: Path, keys: Iterable[str]) -> dict[str, Any]:
    """Keep each candidate/split record; never average across masks or banks."""
    records: list[dict[str, Any]] = []
    for entry in _metric_records(folder):
        path, data = Path(entry["path"]), entry["data"]
        mean = data.get("mean", data.get("metrics", data))
        metrics = {}
        if isinstance(mean, Mapping):
            for key in keys:
                value = mean.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metrics[key] = float(value)
        records.append({
            "path": str(path),
            "candidate": data.get("candidate") or data.get("label") or path.parent.name,
            "split": data.get("split") or path.stem,
            "metrics": metrics,
            "fingerprint": data.get("fingerprint"),
            "bank": data.get("bank") or data.get("bank_identity"),
        })
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_candidate.setdefault(str(record["candidate"]), []).append(record)
    # Compatibility convenience for one-record fixtures; with multiple records
    # this is deliberately None so consumers cannot mistake a pooled mean for a
    # candidate-specific diagnostic.
    components = records[0]["metrics"] if len(records) == 1 else None
    return {"records": records, "by_candidate": by_candidate, "components": components}


def _diagnostics(root: Path, multi: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    exchange_state = _read(root / "exchange" / "state.json")
    initial_receipt = _read(root / "exchange" / "initial.json")
    exchange_rounds = _metric_records(root / "exchange")
    accepted = exchange_state.get("accepted", []) if isinstance(exchange_state, Mapping) else []
    covered = exchange_state.get("covered_directions") if isinstance(exchange_state, Mapping) else None
    if not isinstance(covered, list):
        discovered = []
        seen = set()
        for entry in exchange_rounds:
            data = entry["data"]
            offers = data.get("offers") or data.get("candidates") or data.get("proposals")
            if not isinstance(offers, list):
                continue
            for offer in offers:
                if not isinstance(offer, Mapping):
                    continue
                donor, receiver = offer.get("donor"), offer.get("receiver")
                if donor is None or receiver is None:
                    continue
                key = (donor, receiver)
                if key not in seen:
                    seen.add(key)
                    discovered.append({"donor": donor, "receiver": receiver})
        covered = discovered if discovered else None
    accepted_moves = [
        {key: move.get(key) for key in ("round", "donor", "receiver", "measured_gain", "predicted_gain")}
        for move in accepted if isinstance(move, Mapping)
    ]
    initial_loss = exchange_state.get("initial_loss") if isinstance(exchange_state, Mapping) else None
    if initial_loss is None and isinstance(initial_receipt, Mapping):
        initial_loss = initial_receipt.get("loss")
        if isinstance(initial_loss, Mapping):
            initial_loss = initial_loss.get("AC")
    final_loss = exchange_state.get("loss") if isinstance(exchange_state, Mapping) else None
    ex_cfg = manifest.get("exchange", {}) if isinstance(manifest.get("exchange"), Mapping) else {}
    exchange = dict(
        bank={"id": "legacy scalar ordered 80-pair bank", "path": str(multi.parent.parent / "dlm_context_response50" / "pairs.json")},
        state=dict(exchange_state) if isinstance(exchange_state, Mapping) else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        accepted_moves=accepted_moves,
        covered_directions=covered,
        candidate_round_records=[row["path"] for row in exchange_rounds if "round" in Path(row["path"]).name],
        termination=(exchange_state.get("termination") if isinstance(exchange_state, Mapping) else None),
        warm_start="historical legacy AC physical mask",
        step_d=ex_cfg.get("d", 41),
        proposal_cap=ex_cfg.get("proposals", 8),
        round_cap=ex_cfg.get("rounds", 3),
    )
    return {
        "Multi": _family_diagnostics(multi / "diagnostics", _MULTI_KEYS),
        "Square": _family_diagnostics(root / "metrics" / "Square", _SQUARE_KEYS),
        "Vector": dict(
            _family_diagnostics(root / "metrics" / "Vector", _VECTOR_KEYS),
            vocabulary_normalization="1/V over all emitted vocabulary coordinates",
            pair_reduction="equal pair means",
        ),
        "Exchange": exchange,
    }


def _category_for(row: Mapping[str, Any]) -> str:
    arguments = row.get("arguments", [])
    arg_text = " ".join(str(value) for value in arguments) if isinstance(arguments, (list, tuple)) else str(arguments)
    text = " ".join(str(row.get(key, "")) for key in ("job", "stage", "label", "candidate", "split")) + " " + arg_text
    text = text.lower()
    stage = str(row.get("stage", "")).lower()
    split = str(row.get("split", "")).lower()
    if stage in {"generate", "generation", "gsm8k"} or " generation" in text:
        return "generation"
    if "teacher" in text or stage in {"teachervectors", "teacher_vectors"}:
        return "teacher_setup"
    if "probe" in text:
        return "probes"
    if "diagnostic" in text or split == "diagnostic":
        return "diagnostics"
    if "calibration" in text or "uniform" in text or split == "calibration":
        return "calibration"
    if "exchange" in text or "search" in text:
        return "search"
    if stage in {"scalar_square", "pair_score"}:
        return "diagnostics"
    # A whole-worker candidate receipt combines loading, readout, and task
    # generation; retain it as mixed/other rather than assigning it to one stage.
    return "other"

def _empty_cost_categories() -> dict[str, dict[str, Any]]:
    return {
        category: {"entries": 0, "jobs": [], "forward_calls": None, "candidate_evaluations": None,
                   "gpu_seconds": None, "wall_seconds": None, "peak_memory_bytes": None,
                   "timing": "unknown"}
        for category in _CATEGORIES
    }


def _cost_categories(files: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    categories = _empty_cost_categories()
    for row in files:
        category = _category_for(row)
        out = categories[category]
        out["entries"] += 1
        job = row.get("job") or row.get("worker") or row.get("label")
        if job is not None and job not in out["jobs"]:
            out["jobs"].append(job)
        for field in ("forward_calls", "wall_seconds", "gpu_seconds", "gpu_assigned_seconds"):
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                target = "gpu_seconds" if field == "gpu_assigned_seconds" else field
                out[target] = (out[target] or 0.0) + float(value)
                out["timing"] = "measured"
        for field in ("peak_cuda_bytes", "peak_rss_bytes", "peak_memory_bytes"):
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out["peak_memory_bytes"] = max(out["peak_memory_bytes"] or 0.0, float(value))
        out["candidate_evaluations"] = len(out["jobs"]) if out["jobs"] else None
    for out in categories.values():
        out["jobs"] = sorted(out["jobs"])
    return categories


def _cost_summary(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    # Worker attempt receipts and stage receipts overlap in forward calls. Keep
    # them in separate authoritative/non-additive views rather than summing.
    attempts = [entry["data"] for entry in _metric_records(root / "costs")]
    stages = [entry["data"] for entry in _metric_records(root / "stage_costs")]
    return {
        "planned_nominal": dict(manifest.get("costs", {})) if isinstance(manifest.get("costs"), Mapping) else {},
        "attempts": _cost_categories(attempts),
        "stages": _cost_categories(stages),
        "categories_are_nonadditive": True,
        "historical": {"status": "reused", "timing": "unknown", "gpu_seconds": None, "wall_seconds": None},
        "missing_timing_is_unknown": True,
    }


def _paired(reference: list[Mapping[str, Any]], candidate: list[Mapping[str, Any]], ids: list[int]) -> dict[str, Any]:
    ref = {int(row["example_id"]): bool(row["correct"]) for row in reference}
    cand = {int(row["example_id"]): bool(row["correct"]) for row in candidate}
    gained = sum(not ref[i] and cand[i] for i in ids)
    lost = sum(ref[i] and not cand[i] for i in ids)
    discordant = gained + lost
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(min(gained, lost) + 1)) / (2 ** discordant)
        p = min(1.0, 2.0 * tail)
    return {"gained": gained, "lost": lost, "net": gained - lost, "total": len(ids), "exact_mcnemar_p": p}


def _compatible_ids(reference: Iterable[Mapping[str, Any]], candidate: Iterable[Mapping[str, Any]]) -> list[int]:
    """Return IDs whose identity receipts agree in both arms."""
    fields = ("doc_hash", "prompt_hash", "target_hash", "reference_answer", "input_ids_sha256")
    ref = {int(row["example_id"]): row for row in reference}
    cand = {int(row["example_id"]): row for row in candidate}
    common: list[int] = []
    for eid in sorted(ref.keys() & cand.keys()):
        left, right = ref[eid], cand[eid]
        if all(left.get(field) is not None and right.get(field) is not None and left.get(field) == right.get(field) for field in fields[:-1]) and (left.get(fields[-1]) is None or right.get(fields[-1]) is None or left.get(fields[-1]) == right.get(fields[-1])):
            common.append(eid)
    return common


def _holm(comparisons: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(
        ((key, value["exact_mcnemar_p"]) for key, value in comparisons.items()),
        key=lambda item: (item[1], item[0]),
    )
    running = 0.0
    n = len(ordered)
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * p))
        comparisons[key]["holm_p"] = running


def _scope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "deferred": list(manifest.get("deferred", [])) if isinstance(manifest.get("deferred"), list) else [],
        "limits": [
            "mini100 is repeated development data, including after Holm adjustment",
            "cross-family task differences are end-to-end observations",
            "exact match is not a reasoning or denoising-mechanism annotation",
            "Exchange is a capped old-cost search, not a convergence or C-necessity test",
            "DKD and A-floor remain untested deferred candidates",
        ],
    }


def build_report(root: str | Path, multi: str | Path, rows_by_arm: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Build a complete report from injected rows and on-disk receipts.

    ``rows_by_arm`` may map an arm to a list of validated rows or to a mapping
    containing ``rows`` plus metadata such as ``fingerprint``, ``provenance``,
    ``reused``, or ``alias_of``.  Missing identity receipts never become a
    complete arm.  The returned tuple is ``(JSON-compatible-report, markdown)``.
    """
    root, multi = Path(root), Path(multi)
    manifest = _read(root / "manifest.json", {})
    if not isinstance(manifest, Mapping):
        manifest = {}
    armdefs = manifest.get("arms", {}) if isinstance(manifest.get("arms", {}), Mapping) else {}
    references = manifest.get("references", {}) if isinstance(manifest.get("references", {}), Mapping) else {}
    scores: dict[str, dict[str, Any]] = {}
    row_cache: dict[str, list[Mapping[str, Any]]] = {}
    for label in ALL_LABELS:
        raw = rows_by_arm.get(label, []) if isinstance(rows_by_arm, Mapping) else []
        if (not raw) and isinstance(rows_by_arm, Mapping):
            for alias in {"legacy_uniform": "legacy_Uniform", "legacy_A": "legacy_A", "legacy_AC": "legacy_AC"}.get(label, label),:
                if alias in rows_by_arm:
                    raw = rows_by_arm[alias]
                    break
        rows, injected = _unwrap_rows(raw)
        expected = _expected_ids(injected, manifest)
        valid, invalid = _valid_rows(rows, set(expected))
        definition = _definition_for(label, armdefs, references)
        identity, identity_path, model = _identity_from_files(root, multi, label)
        fp = _fingerprint(injected, identity, model)
        valid_identity = _identity_valid(fp)
        alias_of = injected.get("alias_of")
        if alias_of is None:
            cache = _read((root / "gsm8k" / _candidate_name(label)) / "cache_reuse.json", {})
            if isinstance(cache, Mapping):
                alias_of = cache.get("source_arm") or cache.get("alias_of")
        evaluated = len(valid)
        complete = evaluated == len(expected) and valid_identity and not invalid
        status = "complete" if complete else "partial" if evaluated else "pending"
        if evaluated and not valid_identity:
            status = "invalid_identity"
        provenance = injected.get("provenance")
        if provenance is None:
            provenance = "historical_cached_reference" if label in REFERENCES else "screen_new_or_resumed"
        bank = _bank_metadata(root, multi, label, definition)
        question_links = _question_links(root, multi, label, valid)
        row_cache[label] = valid
        scores[label] = {
            "correct": sum(bool(row["correct"]) for row in valid),
            "evaluated": evaluated,
            "expected": len(expected),
            "accuracy": (sum(bool(row["correct"]) for row in valid) / len(expected)) if complete else None,
            "status": status,
            "complete": complete,
            "identity_valid": valid_identity,
            "identity_status": "validated" if valid_identity else "missing_or_invalid",
            "mask_identity": fp.get("mask_identity") if isinstance(fp, Mapping) else None,
            "sparse_model_sha256": fp.get("sparse_model_sha256") if isinstance(fp, Mapping) else None,
            "fingerprint": dict(fp) if isinstance(fp, Mapping) else None,
            "identity_receipt": _path_text(identity_path),
            "bank": bank,
            "readout": definition.get("readout") or ("legacy-fp32-gold-logodds-fp64-residual" if label not in {"Vector-A", "Vector-AC"} else "centered-emitted-logits-fp64"),
            "allocator": definition.get("allocator") or "historical/cache metadata unavailable",
            "objective_version": definition.get("objective_version"),
            "lambda_response": definition.get("lambda_response"),
            "control": definition.get("control", []),
            "reduction": definition.get("reduction"),
            "edge_contract": definition.get("edge_contract"),
            "provenance": provenance,
            "reused": bool(injected.get("reused", label in REFERENCES or label in {"MS-A", "Short", "Path", "All", "Multi"})),
            "alias_of": alias_of,
            "invalid_rows": invalid,
            "question_files": question_links,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate, reference in CONTRASTS:
        ca, rb = row_cache[candidate], row_cache[reference]
        ids = _compatible_ids(rb, ca)
        record = _paired(rb, ca, ids) if ids else {"gained": 0, "lost": 0, "net": 0, "total": 0, "exact_mcnemar_p": None}
        record.update(
            candidate=candidate,
            reference=reference,
            common_ids=ids,
            common_id_count=len(ids),
            complete=(scores[candidate]["complete"] and scores[reference]["complete"] and len(ids) == 100),
            interpretation="fixed predeclared paired development contrast",
        )
        comparisons[f"{candidate}_vs_{reference}"] = record
    holm_ready = all(row["complete"] for row in scores.values()) and all(row["complete"] for row in comparisons.values())
    if holm_ready:
        _holm(comparisons)

    diagnostics = _diagnostics(root, multi, manifest)
    costs = _cost_summary(root, manifest)
    source = _source_metadata(manifest)
    report: dict[str, Any] = {
        "status": "complete" if holm_ready else "partial",
        "arms": scores,
        "scores": scores,
        "references": {label: scores[label] for label in REFERENCES},
        "comparisons": comparisons,
        "holm_family_size": 7,
        "holm_applied": holm_ready,
        "diagnostics": diagnostics,
        "costs": costs,
        "source_ledger": source,
        "scope": _scope(manifest),
        "development_data": True,
        "no_follow_up_started": True,
        "per_question_outputs": {label: scores[label]["question_files"] for label in ALL_LABELS},
    }
    return report, _markdown(report)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# AC mini100 development screen",
        "",
        "## Hypothesis",
        "Matched within-family response weighting may improve pruning allocation; this remains an empirical screen.",
        "",
        "## Setup",
        "All rows are reported as repeated development data. Bank, readout, allocator, identity, and provenance are shown per arm.",
        "",
        "## Results",
        "| Arm | Correct/evaluated | State | Identity | Mask | Bank/readout | Outputs |",
        "|---|---:|---|---|---|---|---|",
    ]
    for label, row in report["scores"].items():
        identity = "valid" if row["identity_valid"] else "missing/invalid"
        mask = row.get("mask_identity") or "unknown"
        bank = row.get("bank", {}).get("id", "unknown")
        readout = row.get("readout", "unknown")
        outputs = len(row.get("question_files", {}))
        lines.append(f"| {label} | {row['correct']}/{row['evaluated']} | {row['status']} | {identity} | `{mask}` | {bank}; {readout} | {outputs} links |")
    lines += ["", "### Fixed paired contrasts", "", "| Candidate | Reference | Common IDs | Gained | Lost | Exact McNemar p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["comparisons"].values():
        p = "unknown" if row["exact_mcnemar_p"] is None else f"{row['exact_mcnemar_p']:.8g}"
        hp = "not applied" if "holm_p" not in row else f"{row['holm_p']:.8g}"
        lines.append(f"| {row['candidate']} | {row['reference']} | {row['common_id_count']} | {row['gained']} | {row['lost']} | {p} | {hp} |")
    lines += ["", "## Interpretation", "Cross-family differences are end-to-end observations. Scalar and vector objective values use different units. Exact match is not a reasoning annotation. Path has unequal endpoint degrees. Exchange is a capped old-cost search.", "", "## Decision", "Retain positive and negative results; no automatic tuning, full GSM8K, or confirmation phase is started.", "", "## Diagnostics", ""]
    for family, value in report["diagnostics"].items():
        if isinstance(value, Mapping):
            components = value.get("components")
            lines.append(f"- {family}: components={json.dumps(components, sort_keys=True)}")
    lines += ["", "## Costs", "Worker attempt totals and stage breakdowns overlap and are shown separately."]
    for view in ("attempts", "stages"):
        lines += ["", f"### {view.title()}", "", "| Category | Entries | Forward calls | GPU seconds | Wall seconds | Peak memory |", "|---|---:|---:|---:|---:|---:|"]
        for category, value in report["costs"][view].items():
            lines.append(f"| {category} | {value['entries']} | {value['forward_calls'] if value['forward_calls'] is not None else 'unknown'} | {value['gpu_seconds'] if value['gpu_seconds'] is not None else 'unknown'} | {value['wall_seconds'] if value['wall_seconds'] is not None else 'unknown'} | {value['peak_memory_bytes'] if value['peak_memory_bytes'] is not None else 'unknown'} |")
    lines += ["", "Historical imported work is labeled reused; missing historical timings remain unknown.", "", "## Source ledger and scope", f"- Ledger: `{report['source_ledger'].get('ledger')}`", f"- Ledger SHA256: `{report['source_ledger'].get('ledger_sha256')}`"]
    for item in report["scope"]["limits"]:
        lines.append(f"- {item}")
    lines += ["", "## Per-question output links"]
    for label, links in report["per_question_outputs"].items():
        lines.append(f"### {label}")
        for eid, path in links.items():
            lines.append(f"- [{eid}]({path})")
    return "\n".join(lines) + "\n"


__all__ = ["ARMS", "REFERENCES", "CONTRASTS", "build_report"]
