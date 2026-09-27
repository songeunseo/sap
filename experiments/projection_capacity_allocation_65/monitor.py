"""Monitor the authorized tmux pipeline and audit its final artifacts."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.projection_capacity_allocation_65.verify_states import ROOT, write_json


def monitor(pid):
    while True:
        alive = Path(f'/proc/{pid}').exists()
        log = ROOT / 'logs/run.log'
        text = log.read_text() if log.exists() else ''
        events = []
        for line in text.splitlines():
            if line.startswith('{'):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        complete = any(e.get('event') == 'experiment_complete' for e in events)
        status = {'status': 'running' if alive else 'process_exited', 'pid': pid,
                  'completed_projection_curves': len(list((ROOT / 'curves').glob('*.json'))),
                  'total_projection_curves': 224, 'last_event': events[-1] if events else None,
                  'updated_unix_time': time.time()}
        write_json(ROOT / 'status.json', status)
        if complete:
            audit = subprocess.run([sys.executable, '-m', 'experiments.projection_capacity_allocation_65.audit'])
            status['status'] = 'complete_and_audited' if audit.returncode == 0 else 'audit_failed'
            status['audit_exit_code'] = audit.returncode
            write_json(ROOT / 'status.json', status)
            return audit.returncode
        if not alive:
            status['status'] = 'stopped_before_completion'
            status['log_tail'] = text.splitlines()[-20:]
            write_json(ROOT / 'status.json', status)
            return 1
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pid', type=int)
    raise SystemExit(monitor(parser.parse_args().pid))
