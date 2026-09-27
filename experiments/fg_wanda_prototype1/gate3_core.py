"""Frozen Gate-3 helpers for the single Fisher-Geometry prototype."""
import hashlib

import torch
from scipy.stats import binomtest

from experiments.wanda_failure_characterization.core import rowwise_wanda_mask


def raw_mask_sha256(mask):
    return hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def build_standard_and_fg_masks(modules, clean_a, fg_target, target='block_31.ff_out'):
    if target not in modules or target not in clean_a:
        raise KeyError(target)
    standard={}
    for name,module in modules.items():
        score=module.weight.detach().float().abs()*clean_a[name].to(module.weight.device).sqrt()[None,:]
        standard[name]=rowwise_wanda_mask(score,.5)
    fg_target=fg_target.to(standard[target].device)
    if fg_target.shape!=standard[target].shape or fg_target.dtype!=torch.bool:
        raise ValueError('FG target mask shape/dtype mismatch')
    ours=dict(standard);ours[target]=fg_target
    return standard,ours


@torch.no_grad()
def apply_masks(modules,masks):
    if set(modules)!=set(masks):raise ValueError('module/mask key mismatch')
    zeros=0;weights=0
    for name,module in modules.items():
        mask=masks[name].to(module.weight.device)
        expected=module.weight.detach().masked_fill(mask,0)
        module.weight.copy_(expected)
        if not torch.equal(module.weight.eq(0),mask):
            raise RuntimeError(f'applied mask mismatch: {name}')
        zeros+=int(mask.sum());weights+=mask.numel()
    return {'matrix_count':len(modules),'zero_count':zeros,'weight_count':weights,'sparsity':zeros/weights}


def paired_summary(wanda,fg):
    if len(wanda)!=len(fg):raise ValueError('paired sample count differs')
    wc=[bool(x['correct']) for x in wanda];fc=[bool(x['correct']) for x in fg]
    both=sum(a and b for a,b in zip(wc,fc));wo=sum(a and not b for a,b in zip(wc,fc))
    fo=sum(not a and b for a,b in zip(wc,fc));neither=sum(not a and not b for a,b in zip(wc,fc))
    discordant=wo+fo
    p=1.0 if discordant==0 else float(binomtest(min(wo,fo),discordant,.5,alternative='two-sided').pvalue)
    return {'sample_count':len(wc),'both_correct':both,'fg_only':fo,'wanda_only':wo,'both_wrong':neither,
            'wanda_correct':sum(wc),'fg_correct':sum(fc),'correct_delta':fo-wo,
            'delta_percentage_points':100*(fo-wo)/len(wc),'mcnemar_exact_pvalue':p}
