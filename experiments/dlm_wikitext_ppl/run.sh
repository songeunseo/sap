#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/usr/bin/python3 -m experiments.dlm_wikitext_ppl.run prepare
/usr/bin/python3 -m experiments.dlm_wikitext_ppl.run pilot
