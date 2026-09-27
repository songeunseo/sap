#!/usr/bin/env python3
"""Exact tmux-session predicate used by the downstream queue."""
import subprocess
import sys


def exact_session_exists(session_names, target):
    return target in set(session_names)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: queue_control.py SESSION")
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        text=True, capture_output=True, check=False,
    )
    names = result.stdout.splitlines() if result.returncode == 0 else []
    raise SystemExit(0 if exact_session_exists(names, sys.argv[1]) else 1)


if __name__ == "__main__":
    main()
