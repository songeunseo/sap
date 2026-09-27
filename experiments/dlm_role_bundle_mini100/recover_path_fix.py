"""Explicit recovery of the TWO known saved outputs; never generates or rescores."""
import json
import time

from experiments.dlm_role_bundle_mini100 import run
from experiments.dlm_capacity_predictor.audit_existing import audit_prediction_file

BASE_CONFIG = "e1c01d2cdbf4a4f7fb5aa7faa0f95f6bc807d057d0859a66f62e0a802241093d"
OLD_RUN = "fd1a7a8df7d79b0093aef3db1e80bd163d6dd44ca2a49f4128d4e6fff9dc23a9"
SAVED = {
    "add_b0": "a191c1551f6c8a583abbbdbe9ed6b6b94d86354dfbbf4ce4191e690bbd63acee",
    "revert_b0": "559546118affefefe648cdee23d2ea43e991567faa0cca75bb3dfc2df249a199",
}


def main():
    root = run.ROOT
    archive = root / "maintenance/path_fix_20260913"
    c = run.read(root / "config.json")
    if run.sha256(root / "config.json") != BASE_CONFIG or run.sha256(archive / "config.original.json") != BASE_CONFIG:
        raise RuntimeError("original configuration not preserved")
    if run.sha256(archive / "run.original.py") != OLD_RUN:
        raise RuntimeError("original runner not preserved")
    for p, expected in c["source_sha256"].items():
        if p != str(root / "run.py") and run.sha256(p) != expected:
            raise RuntimeError(f"non-runner source changed: {p}")
    amendment = dict(base_config_sha256=BASE_CONFIG, reason="str/Path receipt hash crash after predictions saved; user authorized recovery",
        scientific_configuration_changed=False,
        changed_sources={str(root / "run.py"): dict(before=OLD_RUN, after=run.sha256(root / "run.py"), archive=str(archive / "run.original.py"))},
        new_sources={str(root / p): run.sha256(root / p) for p in ("recover_path_fix.py", "test_receipts.py")})
    run.write(archive / "amendment.json", amendment, frozen=True)
    run.validate()
    for method, expected in SAVED.items():
        path = root / method / "predictions.jsonl"
        if run.sha256(path) != expected:
            raise RuntimeError("saved predictions changed; no recovery")
        audit = audit_prediction_file(path, 100)
        data = run.rows(path)
        run.validate_rows(data, run.rows(c["baselines"]["aggregate"]), c["protocol_hash"])
        if any(row["method"] != method for row in data):
            raise RuntimeError("saved method identity mismatch")
        log = archive / f"{method}.failed.log"
        text = log.read_text()
        events = [json.loads(s) for s in text.splitlines() if s.startswith('{"method":')]
        verifying = [r for r in events if r.get("stage") == "verifying" and r.get("completed") == 100]
        failed = [r for r in events if r.get("stage") == "failed"]
        if len(verifying) != 1 or len(failed) != 1 or failed[0]["error"] != "AttributeError: 'str' object has no attribute 'open'":
            raise RuntimeError("unexpected failure path")
        if 'manifest_sha256=sha256(c["manifests"][method])' not in text:
            raise RuntimeError("failure was not the known post-save hash call")
        if (root / method / "results.json").exists():
            run.completed(method, c)
            continue
        recovery = dict(status="recovered_saved_predictions", predictions_sha256=expected,
            audit_correct=audit["correct"], original_config_sha256=BASE_CONFIG, original_run_sha256=OLD_RUN,
            amendment_sha256=run.sha256(archive / "amendment.json"), failure_log_sha256=run.sha256(log),
            original_generation_reused=True, generated_again=False,
            model_integrity="Original code reached post-save receipt hash after pre/post-model hash equality and row validation. Inferred from code path, not remeasured.",
            unavailable_fields=["sparse_model_sha256 value (not persisted)", "exact evaluator metrics/timing (not persisted)"],
            recovered_at=time.time())
        # Missing original metadata remains explicitly unknown; never fabricate a hash/timer.
        result = run.result_metadata(method, c, data, None,
            {"accuracy": audit["accuracy"], "num_examples": 100, "full_num_examples": 1319,
             "eval_seconds_unavailable": True}, "0" if method.startswith("add") else "1")
        result.update(predictions_sha256=expected, recovered_after_receipt_failure=True,
                      recovery_evidence=recovery)
        run.write(root / method / "recovery.json", recovery, frozen=True)
        run.write(root / method / "results.json", result, frozen=True)
        run.completed(method, c)
        run.event(method, "complete", correct=audit["correct"], total=100, recovered_saved_predictions=True)
    print("Recovery complete; original predictions and scientific config unchanged; no generation performed.")


if __name__ == "__main__": main()
