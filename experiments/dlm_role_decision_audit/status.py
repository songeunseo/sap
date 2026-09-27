#!/usr/bin/env python3
import json, re, time
from pathlib import Path
import torch

ROOT=Path("experiments/dlm_role_decision_audit"); RUN=Path("/DATA/tmluser1/sap-dlm-role-decision-audit")
for bg in ("dense","role_sparse"):
    path=RUN/f"bundle_{bg}.pt"; done=0
    if path.exists(): done=len(torch.load(path,map_location="cpu",weights_only=False).get("rows",[]))
    total=2400
    log=ROOT/"logs"/f"{bg}.log"; eta="unknown"
    if log.exists():
        events=[]
        for line in log.read_text(errors="ignore").splitlines()[-300:]:
            try:
                x=json.loads(line)
                if x.get("event")=="bundle_condition": events.append(x)
            except: pass
        if len(events)>2 and done<total:
            rate=(events[-1]["time"]-events[0]["time"])/max(events[-1]["completed"]-events[0]["completed"],1)
            eta=f"{(total-done)*rate/3600:.2f}h"
    print(f"{bg}: {done}/{total}, ETA {eta}")
full=RUN/"full_allocations.pt"
print(f"full allocations: {len(torch.load(full,map_location='cpu',weights_only=False).get('rows',[])) if full.exists() else 0}/120")
print("analysis:", "complete" if (ROOT/"analysis.json").exists() else "pending")
