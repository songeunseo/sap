"""Figure 1: exact constructed residuals, not model outputs or predictions."""
from fractions import Fraction
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGES = {d: [(j, j+d) for j in range(8) if not j & d] for d in (1, 2, 4)}
EXAMPLES = {
    'constant': [1, 1, 1, 1, 1, 1, 1, 1],
    'changing': [1, 1, -1, -1, 1, 1, -1, -1],
}


def exact_losses(residuals):
    a = sum(Fraction(e*e, 8) for e in residuals)
    scales = {d: sum(Fraction((residuals[k]-residuals[j])**2, 4) for j, k in pairs)
              for d, pairs in EDGES.items()}
    natural = sum(scales.values()) / 3
    return {'A': a, **{f'C{d}': v for d, v in scales.items()},
            'C_natural': natural, 'L_Multi': a + natural}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    losses = {name: exact_losses(values) for name, values in EXAMPLES.items()}
    assert losses['constant'] == dict(A=1, C1=0, C2=0, C4=0, C_natural=0, L_Multi=1)
    assert losses['changing'] == dict(A=1, C1=0, C2=4, C4=0, C_natural=Fraction(4, 3), L_Multi=Fraction(7, 3))
    output = HERE / 'figures'
    output.mkdir(exist_ok=True)
    payload = {
        'kind': 'constructed_algebraic_illustration',
        'contains_empirical_measurements': False,
        'source_design': '/home/tmluser1/sap/research/ac_multiscale_monotone_design_2026-09-22.md#43',
        'phase_pairs': EDGES,
        'examples': {name: {'residuals': EXAMPLES[name],
                           'exact_losses': {k: str(v) for k, v in metrics.items()}}
                     for name, metrics in losses.items()},
    }
    (output / 'response-example.json').write_text(json.dumps(payload, indent=2) + '\n')

    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1.12, 1]})
    colors = {'constant': '#0072B2', 'changing': '#D55E00'}
    styles = {'constant': ('o', '--'), 'changing': ('s', '-')}
    ax = axes[0]
    for name, residuals in EXAMPLES.items():
        marker, linestyle = styles[name]
        ax.plot(range(8), residuals, color=colors[name], marker=marker,
                linestyle=linestyle, linewidth=2, markersize=6,
                markerfacecolor='white' if name == 'constant' else colors[name],
                label=f'{name.capitalize()} residual', zorder=3 if name == 'constant' else 2)
    ax.axhline(0, color='#999999', linewidth=.7, zorder=0)
    ax.set(xticks=range(8), yticks=[-1, 0, 1], ylim=(-1.4, 1.6),
           xlabel='Reveal phase j (more visible context →)',
           ylabel='Sparse − dense log-odds')
    ax.set_title('(a) Same error magnitude at every state', loc='left', fontsize=11, pad=12)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=.18)

    ax = axes[1]
    keys = ['A', 'C1', 'C2', 'C4', 'C_natural']
    for offset, name in [(-.18, 'constant'), (.18, 'changing')]:
        xs = [i+offset for i in range(len(keys))]
        vals = [float(losses[name][key]) for key in keys]
        ax.bar(xs, vals, width=.32, color=colors[name],
               hatch='///' if name == 'constant' else None, alpha=.9)
        for x, value, key in zip(xs, vals, keys):
            if value == 0:
                ax.plot(x, 0, marker='_', color=colors[name], markersize=9, markeredgewidth=2, clip_on=False)
            ax.text(x, value+.12, str(losses[name][key]), ha='center', va='bottom', fontsize=9)
    ax.set(xticks=range(5), xticklabels=['$A$', '$C_1$', '$C_2$', '$C_4$', '$C_{natural}$'],
           ylim=(0, 4.65), yticks=[0, 1, 2, 3, 4], ylabel='Loss (squared log-odds)')
    ax.set_title('(b) Pairing changes the response penalty', loc='left', fontsize=11, pad=12)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.23, top=.85, wspace=.31)
    fig.text(.5, .045, 'Constructed illustration • no model measurements', ha='center', color='#555555', fontsize=10)
    fig.savefig(output / 'response-example.png', dpi=220)
    fig.savefig(output / 'response-example.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(json.dumps(payload['examples'], indent=2))


if __name__ == '__main__':
    main()
