"""Corrective rerun. V1 artifacts remain immutable; final data were already seen."""
import json
from pathlib import Path
import numpy as np
from experiments.dlm_role_exchange_prediction.core import atomic_json, sha256, json_digest

ROOT = Path('experiments/dlm_role_exchange_prediction_v2')
OLD = Path('experiments/dlm_role_exchange_prediction')
STORE = Path('/DATA/tmluser1/sap-dlm-role-exchange-prediction-v2')
OLD_STORE = Path('/DATA/tmluser1/sap-dlm-role-exchange-prediction')

def ridge(x, y, test, weights=None, alpha=1.):
    """min weighted mean squared error + alpha*||beta||² (no intercept penalty)."""
    x, y, test = np.asarray(x,float), np.asarray(y,float), np.asarray(test,float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights,float)
    w = w/w.sum()
    mean = np.sum(x*w[:,None],axis=0)
    scale = np.sqrt(np.sum((x-mean)**2*w[:,None],axis=0))
    scale[scale < 1e-12] = 1.
    a = np.column_stack([np.ones(len(x)),(x-mean)/scale])
    b = np.column_stack([np.ones(len(test)),(test-mean)/scale])
    penalty = np.eye(a.shape[1])*alpha; penalty[0,0] = 0
    beta = np.linalg.solve(a.T@(a*w[:,None])+penalty,a.T@(w*y))
    return b@beta, {'intercept':float(beta[0]),'beta':beta[1:].tolist(),
                    'mean':mean.tolist(),'scale':scale.tolist(),'objective':'weighted MSE + alpha norm2'}

def equal_layer_weights(rows):
    keys = [(r['document'],r['layer']) for r in rows]
    counts = {k:keys.count(k) for k in set(keys)}
    return np.array([1./counts[k] for k in keys])

def bootstrap_cells(values, rows, seed=20260912, family=3):
    """Equal document/layer estimate; independent resampling of documents and 4-layer blocks.
    Conditional on fitted OOF models, not a refit bootstrap.
    """
    docs = sorted({r['document'] for r in rows})
    groups = sorted({r['layer']//4 for r in rows})
    cell = np.empty((len(docs),len(groups)))
    values = np.asarray(values,float)
    for di,d in enumerate(docs):
        for gi,g in enumerate(groups):
            layers = sorted({r['layer'] for r in rows if r['document']==d and r['layer']//4==g})
            if not layers: raise ValueError('empty bootstrap cell')
            cell[di,gi] = np.mean([values[[i for i,r in enumerate(rows) if r['document']==d and r['layer']==l]].mean() for l in layers])
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(20000):
        samples.append(cell[np.ix_(rng.integers(len(docs),size=len(docs)),rng.integers(len(groups),size=len(groups)))].mean())
    q=.05/(2*family)
    return {'mean':float(cell.mean()),'simultaneous_ci':np.quantile(samples,[q,1-q]).tolist(),
            'resamples':20000,'family':family,'conditional_on_fitted_models':True}

def validate_sources():
    cfg=json.loads((ROOT/'config.json').read_text())
    for path,digest in cfg['source_hashes'].items():
        if sha256(path)!=digest: raise RuntimeError('Changed frozen input: '+path)
    for path,digest in cfg['code_hashes'].items():
        if sha256(path)!=digest: raise RuntimeError('Changed frozen code: '+path)
    return cfg
