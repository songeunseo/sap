"""CPU-only descriptive audit of existing A+C probes; never loads a model."""
import json
import math
from pathlib import Path

import torch


def main():
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'experiments/dlm_context_response50'

    def read(path):
        return json.loads(path.read_text())

    def load(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    baseline = load(root / 'uniform_errors.pt')
    metadata = [read(root / 'probes' / f'block{b:02d}.json') for b in range(32)]
    lower = [load(root / 'probes' / f'block{b:02d}_0.48_errors.pt') for b in range(32)]
    upper = [load(root / 'probes' / f'block{b:02d}_0.52_errors.pt') for b in range(32)]

    def embed(errors, metric):
        parts = []
        for error in errors:
            error = error.double()
            count = error.shape[1]
            parts.append(error.flatten() / math.sqrt(len(errors) * 2 * count))
            if metric == 'AC':
                parts.append((error[1] - error[0]) / math.sqrt(len(errors) * count))
        return torch.cat(parts)

    report = {
        'analysis': 'Saved probe linear-response reconstruction; descriptive; no new forwards or optimized masks',
        'methods': {},
    }
    for metric in ('A', 'AC'):
        residual = embed(baseline, metric)
        jacobian = torch.stack([
            (embed(upper[b], metric) - embed(lower[b], metric)) /
            (metadata[b]['conditions']['0.52']['pruned'] - metadata[b]['conditions']['0.48']['pruned'])
            for b in range(32)
        ], dim=1)
        gram = jacobian.T @ jacobian
        norms = gram.diag().sqrt()
        cosines = gram / (norms[:, None] * norms[None, :])
        off_diagonal = cosines[~torch.eye(32, dtype=torch.bool)]
        result = {
            'baseline_recomputed': residual.square().sum().item(),
            'offdiag_cosine_mean': off_diagonal.mean().item(),
            'offdiag_cosine_min': off_diagonal.min().item(),
            'offdiag_cosine_max': off_diagonal.max().item(),
            'candidate_predictions': {},
        }
        for method in ('A', 'AC'):
            entries = read(root / method / 'mask_manifest.json')['entries']
            blocks = [[e for e in entries if e['name'].startswith(f'block_{b:02d}.')] for b in range(32)]
            delta = torch.tensor([
                sum(e['selected_mask']['pruned'] - 0.5 * e['weights'] for e in block)
                for block in blocks
            ], dtype=torch.float64)
            diagonal = (delta.square() * gram.diag()).sum().item()
            result['candidate_predictions'][method] = {
                'actual': read(root / method / 'joint_distortion.json')['mean'][metric],
                'linear_response_pred': (residual + jacobian @ delta).square().sum().item(),
                'linear_term': (2 * residual @ (jacobian @ delta)).item(),
                'diagonal_quadratic': diagonal,
                'cross_layer_quadratic': (delta @ gram @ delta).item() - diagonal,
                'max_abs_delta_fraction': max(
                    abs(delta[b].item()) / sum(e['weights'] for e in blocks[b])
                    for b in range(32)
                ),
            }
        report['methods'][metric] = result
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
