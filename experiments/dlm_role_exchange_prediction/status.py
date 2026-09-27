#!/usr/bin/env python3
import json
from pathlib import Path
import torch

R=Path("/DATA/tmluser1/sap-dlm-role-exchange-prediction")
for split in ("development","final"):
    p=R/f"{split}.pt"; done=0
    if p.exists(): done=int(torch.load(p,map_location="cpu",weights_only=False).get("completed_states",0))
    eta="unknown"; log=Path("experiments/dlm_role_exchange_prediction/logs")/f"{split}.log"
    if log.exists():
        events=[]
        for line in log.read_text(errors="ignore").splitlines():
            try:
                row=json.loads(line)
                if row.get("event")=="state_complete": events.append(row)
            except Exception: pass
        if events and done<80:
            seconds=float(events[-1]["elapsed"])/int(events[-1]["state"])
            eta=f"{seconds*(80-done)/60:.1f}m"
    print(f"{split}: {done}/80, ETA {eta}")
print("analysis:","complete" if Path("experiments/dlm_role_exchange_prediction/analysis.json").exists() else "pending")
