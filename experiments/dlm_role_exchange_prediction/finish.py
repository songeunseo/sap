#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path
import torch

runtime=Path("/DATA/tmluser1/sap-dlm-role-exchange-prediction")
while True:
    complete=True
    for split in ("development","final"):
        path=runtime/f"{split}.pt"
        if not path.exists() or int(torch.load(path,map_location="cpu",weights_only=False).get("completed_states",0))<80:
            complete=False
    if complete: break
    time.sleep(30)
subprocess.run(["/usr/bin/python3","-u","experiments/dlm_role_exchange_prediction/analyze.py"],check=True)
