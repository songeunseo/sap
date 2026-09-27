"""Exact-k LLaDA Eq.6 estimator; no causal shift or fixed-timestep surrogate."""
import hashlib
import json
import math
import re

import torch
import torch.nn.functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def articles(lines):
    """Keep WikiText top-level article boundaries, including nested headings."""
    result, current, title = [], [], None
    for line in lines:
        match = re.fullmatch(r'\s*=\s+([^=]+?)\s+=\s*', line)
        if match:
            if current:
                result.append(dict(title=title, text='\n'.join(current)))
            title, current = match.group(1), [line]
        else:
            current.append(line)
    if current:
        result.append(dict(title=title, text='\n'.join(current)))
    return [x for x in result if x['text'].strip()]


def block_seed(seed, block_id):
    return int(digest([seed, block_id])[:15], 16)


def masks(length, count, seed):
    if length < 1 or count < 1:
        raise ValueError('positive target length and MC count required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(count):
        k = int(torch.randint(1, length + 1, (), generator=generator))
        mask = torch.zeros(length, dtype=torch.bool)
        mask[torch.randperm(length, generator=generator)[:k]] = True
        yield mask


def mask_digest(length, count, seed):
    h = hashlib.sha256()
    for mask in masks(length, count, seed):
        h.update(mask.numpy().tobytes())
    return h.hexdigest()


def normalized_nelbo(logits, clean, mask):
    """(L/k)*sum CE divided by L = mean CE over exactly k masked tokens."""
    if logits.ndim != 2 or clean.shape != mask.shape or logits.shape[0] != len(clean):
        raise ValueError('same-position logits/labels required')
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError('nonempty explicit boolean corruption mask required')
    value = F.cross_entropy(logits[mask].float(), clean[mask], reduction='mean')
    if not torch.isfinite(value):
        raise ValueError('nonfinite loss')
    return value


def summarize(rows):
    if not rows:
        raise ValueError('no evaluated blocks')
    tokens = sum(r['tokens'] for r in rows)
    mean = sum(r['tokens'] * r['token_nelbo'] for r in rows) / tokens
    variance = sum((r['tokens'] / tokens)**2 * r['mc_variance'] / r['mc_samples'] for r in rows)
    return dict(token_nelbo=mean, ppl_upper_bound_estimate=math.exp(mean),
                mc_standard_error=math.sqrt(variance), tokens=tokens, blocks=len(rows),
                note='Monte Carlo estimate of a bound; finite-sample estimate is not a guaranteed upper bound; SE is MC noise, not corpus generalization uncertainty')
