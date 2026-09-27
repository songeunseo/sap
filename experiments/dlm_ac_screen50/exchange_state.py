"""Recover a committed exchange round before doing additional model work."""
from pathlib import Path
from .integrity import read_checked,write_checked,freeze_checked


def recover(root,state):
    root=Path(root)
    while not state['done']:
        path=root/f'round{state["round"]}.json'
        if not path.exists():break
        journal=read_checked(path)
        if journal['before']!=state:raise RuntimeError('Exchange round does not match incumbent')
        after=journal['after']
        if after['config_sha256']!=state['config_sha256'] or after['round']!=state['round']+1 or after['evaluations']<state['evaluations'] or after['evaluations']>24:raise RuntimeError('Invalid committed exchange transition')
        write_checked(root/'state.json',after);state=after
    return state


def commit(root,before,offers,after):
    root=Path(root)
    freeze_checked(root/f'round{before["round"]}.json',dict(before=before,offers=offers,after=after))
    write_checked(root/'state.json',after)
    return after
