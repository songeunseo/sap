"""Explicit, hash-verified relocation. No unrelated caches or system files touched."""
import json
import os
import shutil
from pathlib import Path
from experiments.dlm_dual_role_allocation.io import file_sha256 as sha, atomic_write_json as write
ROOT=Path('/home/tmluser1/sap/experiments')
DEST=Path('/DATA/tmluser1/sap_storage_recovery_20260914')

def inventory(path):
    return {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}

def main():
    plans=[(ROOT/'dlm_capacity_predictor/runtime',DEST/'dlm_capacity_predictor/runtime')]
    plans += [(ROOT/'dlm_allocation_baselines65'/x,DEST/'dlm_allocation_baselines65'/x)
              for x in ('alpha','dlp','dsa','lsa','logs')]
    DEST.mkdir(parents=True,exist_ok=True)
    receipt=DEST/'migration.json'
    records=json.loads(receipt.read_text()) if receipt.exists() else []
    for source,target in plans:
        if source.is_symlink():
            if source.resolve()!=target:raise RuntimeError('unexpected symlink')
            continue
        if target.exists():raise RuntimeError('destination exists; inspect before retry')
        before=inventory(source);target.parent.mkdir(parents=True,exist_ok=True)
        # Copy then verify, before deleting *only* the validated original files.
        shutil.copytree(source,target,symlinks=True)
        if inventory(target)!=before or inventory(source)!=before:raise RuntimeError('migration content mismatch; originals retained')
        for p in source.rglob('*'):
            if p.is_symlink():raise RuntimeError('unexpected internal symlink; originals retained')
        records.append(dict(source=str(source),target=str(target),hashes=before,status='copied_verified'))
        write(receipt,records)
        shutil.rmtree(source)
        source.symlink_to(target,target_is_directory=True)
        if inventory(source)!=before:raise RuntimeError('linked content mismatch')
        records[-1]['status']='moved_verified_linked';write(receipt,records)
        print(f'moved and verified: {source} -> {target}',flush=True)

if __name__=='__main__':main()
