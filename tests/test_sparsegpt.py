import torch
from torch import nn

from lib.sparsegpt import SparseGPT


def test_token_weights_broadcast_over_features_once_before_hessian_product():
    sparsegpt = SparseGPT(nn.Linear(2, 2, bias=False))
    inputs = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    token_weights = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    sparsegpt.add_batch(inputs, torch.empty(0), token_weights=token_weights)

    torch.testing.assert_close(
        sparsegpt.H,
        torch.tensor([[1710.0, 1996.0], [1996.0, 2336.0]]),
    )


def test_all_ones_token_weights_reproduce_plain_hessian():
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    plain = SparseGPT(nn.Linear(2, 2, bias=False))
    weighted = SparseGPT(nn.Linear(2, 2, bias=False))

    plain.add_batch(inputs, torch.empty(0))
    weighted.add_batch(inputs, torch.empty(0), token_weights=torch.ones(1, 2))

    torch.testing.assert_close(weighted.H, plain.H, rtol=0, atol=0)
