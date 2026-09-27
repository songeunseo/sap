"""Show live progress for the two 50% projection-allocation experiments."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print('DLM PPL50 projection | ' + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    print(f'{"Method":18} {"Stage":24} {"Progress":20} {"NELBO":>10} {"PPL bound":>12}')
    print('-' * 90)
    for method in ('owl_projection', 'dsa_projection'):
        folder = ROOT / method
        result = folder / 'validation/results.json'
        progress = folder / 'progress.json'
        if result.exists():
            row = json.loads(result.read_text())['summary']
            print(f'{method:18} {"complete":24} {"551/551 chunks":20} {row["token_nelbo"]:10.6f} {row["ppl_upper_bound_estimate"]:12.6f}')
        elif progress.exists():
            row = json.loads(progress.read_text())
            stage = row['stage']
            if stage == 'evaluating_validation':
                detail = f'{row["blocks_completed"]}/{row["blocks_total"]} chunks'
            elif stage == 'calibration':
                detail = f'block {row["block"]}: {row["completed"]}/{row["total"]}'
            elif 'completed' in row:
                detail = f'{row["completed"]}/{row["total"]}'
            else:
                detail = row.get('error', '-')[:20]
            print(f'{method:18} {stage:24} {detail:20} {"-":>10} {"-":>12}')
        else:
            print(f'{method:18} {"queued":24} {"-":20} {"-":>10} {"-":>12}')


if __name__ == '__main__':
    main()
