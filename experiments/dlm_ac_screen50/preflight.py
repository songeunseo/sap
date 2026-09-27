"""CPU-only physical checkpoint and storage contracts."""
import json
import shutil
from pathlib import Path
from experiments.dlm_multiscale_ac50.artifacts import sha
DENSE_MODEL_SHA256='2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc'

def checkpoint_contract(config,refs):
    headers={};hashes={}
    for name,size in sorted(config['checkpoint_files'].items()):
        p=Path(name)
        if p.stat().st_size!=size:raise ValueError('Checkpoint byte size changed: '+name)
        with p.open('rb') as f:
            n=int.from_bytes(f.read(8),'little');headers[name]=json.loads(f.read(n))
        hashes[name]=sha(p)
    mapping={}
    for r in refs:
        block,projection=r['name'].split('.')
        suffix=f'.blocks.{int(block.split("_")[1])}.{projection}.weight'
        matches=[(p,k,v) for p,h in headers.items() for k,v in h.items() if k.endswith(suffix)]
        if len(matches)!=1:raise ValueError('Nonunique checkpoint key: '+r['name'])
        p,k,v=matches[0]
        if v['shape']!=r['shape'] or v['dtype']!='BF16':raise ValueError('Checkpoint shape/dtype mismatch: '+r['name'])
        mapping[r['name']]=dict(path=p,key=k,shape=v['shape'],dtype=v['dtype'])
    if len(mapping)!=224:raise ValueError('Expected 224 physical checkpoint keys')
    return dict(checkpoint_map=mapping,checkpoint_sha256=hashes,dense_model_sha256=DENSE_MODEL_SHA256)

def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Cannot measure available host RAM')

def storage_check(root,memory,free_disk=None,free_ram=None):
    disk=shutil.disk_usage(root).free if free_disk is None else free_disk
    ram=available_ram() if free_ram is None else free_ram
    if disk<memory['total_teacher_bytes']+2**30:raise RuntimeError('Insufficient disk for both teachers')
    if ram<memory['estimated_pair_ram_bytes']+2**30:raise RuntimeError('Insufficient RAM for exact full-vocabulary pair reduction')
    return dict(free_disk_bytes=disk,available_ram_bytes=ram,headroom_bytes=2**30,reservation='per-bank posix_fallocate before first forward; allocated filesystem blocks prevent concurrent overcommit',fallback='none')
