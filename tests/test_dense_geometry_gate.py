import torch
from experiments.wanda_failure_characterization.geometry_core import rms_forward, rms_jvp, score_directions


def test_rms_directional_derivative_matches_forward_autograd():
    h = torch.tensor([[1., -2., 3.], [2., 1., -1.]], dtype=torch.float64)
    d = torch.tensor([[.1, .2, -.3], [.4, -.2, .1]], dtype=torch.float64)
    g = torch.tensor([1., 2., .5], dtype=torch.float64)
    _, actual = torch.func.jvp(lambda x: rms_forward(x, g, 1e-5), (h,), (d,))
    assert torch.allclose(rms_jvp(h, d, g, 1e-5), actual, atol=1e-12, rtol=1e-12)


def test_streamed_geometry_matches_autograd_scalar_and_ignores_logit_translation():
    h = torch.tensor([[1., -2., 3.], [2., 1., -1.]])
    d = torch.tensor([[[.1, .2, -.3], [.4, -.2, .1]]])
    g = torch.tensor([1., 2., .5])
    w = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    def f(x):
        return rms_forward(x, g, 1e-5) @ w.T
    z, v = torch.func.jvp(f, (h,), (d[0],))
    p = z.softmax(-1)
    expected = .5 * ((p * v.square()).sum(-1) - (p * v).sum(-1).square()).mean()
    got = score_directions(h, d, g, 1e-5, w, vocab_chunk=2)[0]
    assert torch.allclose(got[2].float(), expected, atol=1e-7)
    assert torch.allclose(got[1].float(), v.square().sum(-1).mean(), atol=1e-6)
    shifted = score_directions(h, d, g, 1e-5, w + torch.tensor([.2, -.1, .3]), vocab_chunk=2)[0]
    assert torch.allclose(got[2], shifted[2], atol=1e-7)
    zero = score_directions(h, torch.zeros_like(d), g, 1e-5, w, vocab_chunk=2)
    assert torch.equal(zero, torch.zeros_like(zero))
