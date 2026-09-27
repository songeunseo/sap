import copy
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from model import LLaDAConfig

from experiments.dlm_loss_aggregation.run import build_calibration_manifest, historical_state_digest, load_config
from lib.data import get_loaders


ROOT = Path(__file__).parent


def main():
    config = copy.deepcopy(load_config("experiments/dlm_loss_aggregation/config.yaml"))
    config["calibration"].update({"seed": 1, "sequence_indices": list(range(8, 16)),
                                  "timesteps": [.1, .3, .5, .7, .9]})
    model_config = LLaDAConfig.from_pretrained(config["model"]["id"], revision=config["model"]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True)
    loader, _ = get_loaders("wikitext2", nsamples=8, seed=1, seqlen=256, tokenizer=tokenizer)
    clean = [sample[0] for sample in loader]
    document = build_calibration_manifest(clean, model_config.mask_token_id, config)
    prior = json.loads(Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json").read_text())
    old = {hashlib.sha256(json.dumps(row["clean_ids"], separators=(",", ":")).encode()).hexdigest() for row in prior["states"][:8]}
    new = {hashlib.sha256(json.dumps(row["clean_ids"], separators=(",", ":")).encode()).hexdigest() for row in document["states"][:8]}
    if old & new:
        raise RuntimeError("held-out clean spans overlap calibration spans")
    document["mask_id"] = model_config.mask_token_id
    document["historical_state_sha256"] = historical_state_digest(document)
    document["preregistered_primary_targets"] = ["Y50_KL", "Y50_pos"]
    document["frozen_before_failure_map"] = True
    temporary = ROOT / "heldout_state_manifest.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(ROOT / "heldout_state_manifest.json")
    print(document["historical_state_sha256"])


if __name__ == "__main__":
    main()
