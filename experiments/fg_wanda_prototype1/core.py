"""Single frozen prototype: Fisher diagonal times MLP input energy."""
import torch
import torch.nn.functional as F
from experiments.wanda_failure_characterization.geometry_core import rms_forward


@torch.no_grad()
def kappa_diagonal(h, gamma, eps, weight, chunk=8192):
    """Exact moment identity; FP32 continuous logits and FP64 softmax/moments.

    Streams existing head vocabulary rows. No Jacobian or square curvature matrix.
    """
    h, gamma = h.float(), gamma.float()
    y = rms_forward(h, gamma, eps)
    z = torch.cat([F.linear(y, weight[start:start+chunk].float())
                   for start in range(0, weight.shape[0], chunk)], -1)
    p = z.double().softmax(-1)
    hd, gd = h.double(), gamma.double()
    radius = (hd.square().mean(-1, keepdim=True) + eps).sqrt()
    g = z.double() * radius
    mu_w = torch.zeros_like(hd); second_w = torch.zeros_like(hd); wg = torch.zeros_like(hd)
    for start in range(0, weight.shape[0], chunk):
        w = weight[start:start+chunk].double()
        pc = p[:,start:start+chunk]
        mu_w += pc @ w
        second_w += pc @ w.square()
        wg += (pc*g[:,start:start+chunk]) @ w
    mu_g = (p*g).sum(-1,keepdim=True)
    var_g = (p*g.square()).sum(-1,keepdim=True)-mu_g.square()
    var_w = second_w-mu_w.square()
    cov = wg-mu_w*mu_g
    alpha = gd/radius
    beta = hd/(h.shape[-1]*radius.pow(3))
    t1=alpha.square()*var_w; t2=beta.square()*var_g; t3=2*alpha*beta*cov
    raw=t1+t2-t3
    tolerance=1e-12+1e-9*(t1.abs()+t2.abs()+t3.abs())
    audit={'minimum':raw.min().item(),'negative_count':int((raw<0).sum()),
           'negative_minimum':raw[raw<0].min().item() if (raw<0).any() else None,
           'material_negative_count':int((raw < -tolerance).sum()),'values':raw.numel(),
           'max_tolerance':tolerance.max().item()}
    if audit['material_negative_count']:
        raise RuntimeError('material negative kappa: '+str(audit))
    return raw.clamp_min(0),audit
