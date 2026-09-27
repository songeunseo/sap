import torch
from types import SimpleNamespace
from experiments.fg_wanda_prototype1.core import kappa_diagonal
from experiments.fg_wanda_prototype1.run import evaluate_target_full
from experiments.fg_wanda_prototype1.gate3_core import (
    apply_masks,
    build_standard_and_fg_masks,
    paired_summary,
)
from experiments.wanda_failure_characterization.geometry_core import rms_forward


def test_kappa_identity_equals_basis_jvp_without_factor_half():
    h=torch.tensor([[1.,-2.,3.],[3.,1.,-1.]])
    gamma=torch.tensor([1.,.5,2.])
    w=torch.tensor([[.4,.1,-.2],[-.2,.8,.1],[.3,-.1,.9],[.1,.2,.3]])
    def f(x): return rms_forward(x,gamma,1e-5)@w.T
    p=f(h).double().softmax(-1)
    expected=[]
    for i in range(3):
        d=torch.zeros_like(h);d[:,i]=1
        _,v=torch.func.jvp(f,(h,),(d,))
        v=v.double()
        expected.append((p*v.square()).sum(-1)-(p*v).sum(-1).square())
    k,audit=kappa_diagonal(h,gamma,1e-5,w,chunk=2)
    assert audit['material_negative_count']==0
    assert torch.allclose(k,torch.stack(expected,-1),atol=1e-7,rtol=1e-5)


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_out = torch.nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.ff_out(x)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(31)] + [_ToyBlock()])
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = blocks

    def forward(self, x):
        return SimpleNamespace(logits=self.model.transformer.blocks[31](x.float()))


def test_full_path_variants_share_one_batch_and_apply_only_requested_mask():
    model = _ToyModel()
    model.model.transformer.blocks[31].ff_out.weight.data.copy_(
        torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    )
    noisy = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]]])
    mask = torch.tensor([[False, True, False], [True, False, False]])

    logits, reconstruction = evaluate_target_full(model, noisy, [None, mask])

    weight = model.model.transformer.blocks[31].ff_out.weight
    expected_dense = torch.nn.functional.linear(noisy[0], weight)
    expected_masked = torch.nn.functional.linear(noisy[0], weight.masked_fill(mask, 0))
    assert logits.shape == (2, 2, 2)
    assert torch.equal(logits[0], expected_dense)
    assert torch.equal(logits[1], expected_masked)
    assert reconstruction[0]['relative_squared_error'] == 0.0
    assert reconstruction[1]['relative_squared_error'] > 0.0


def test_gate3_uses_standard_masks_everywhere_except_the_frozen_fg_target():
    modules = {
        'block_00.q_proj': torch.nn.Linear(4, 2, bias=False),
        'block_31.ff_out': torch.nn.Linear(4, 2, bias=False),
    }
    for module in modules.values():
        module.weight.data.copy_(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]))
    clean_a = {name: torch.ones(4) for name in modules}
    fg_target = torch.tensor([[False, False, True, True], [True, True, False, False]])

    standard, ours = build_standard_and_fg_masks(modules, clean_a, fg_target)

    assert torch.equal(standard['block_00.q_proj'], ours['block_00.q_proj'])
    assert torch.equal(ours['block_31.ff_out'], fg_target)
    assert not torch.equal(standard['block_31.ff_out'], fg_target)

    summary = apply_masks(modules, ours)
    assert summary['matrix_count'] == 2
    assert summary['sparsity'] == 0.5
    for name, module in modules.items():
        assert torch.equal(module.weight.eq(0), ours[name])


def test_paired_summary_reports_direction_and_exact_two_sided_binomial():
    wanda = [{'correct': True}, {'correct': True}, {'correct': False}, {'correct': False}]
    ours = [{'correct': True}, {'correct': False}, {'correct': True}, {'correct': True}]
    result = paired_summary(wanda, ours)
    assert result['both_correct'] == 1
    assert result['wanda_only'] == 1
    assert result['fg_only'] == 2
    assert result['both_wrong'] == 0
    assert result['correct_delta'] == 1
    assert result['mcnemar_exact_pvalue'] == 1.0
