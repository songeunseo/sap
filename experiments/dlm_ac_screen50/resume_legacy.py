"""Stage-one resume of frozen candidates while the new screen is implemented."""
import os
import signal
import time
from pathlib import Path
from experiments.dlm_multiscale_ac50.artifacts import DEFAULT_ROOT, lock, write
from experiments.dlm_multiscale_ac50.prepare import validate
from experiments.dlm_multiscale_ac50.run import run_jobs

ROOT=Path(__file__).resolve().parent/'output'
def main():
    if not os.environ.get('TMUX'): raise RuntimeError('Requires tmux')
    def stop(signum,frame): raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    started=time.time()
    with lock(ROOT/'screen.lock'), lock(DEFAULT_ROOT/'pipeline.lock'):
        validate(DEFAULT_ROOT)
        write(ROOT/'legacy_resume.json',dict(status='running',pid=os.getpid(),gpu='3',started=started,scope=['Short','Multi','A','Path','All']))
        try:
            run_jobs(DEFAULT_ROOT,[(f'development_{a}',['--job','candidate','--method',a,'--split','development']) for a in ['Short','Multi','A','Path','All']],['3'])
            write(ROOT/'legacy_resume.json',dict(status='complete',started=started,ended=time.time()))
        except BaseException as e:
            write(ROOT/'legacy_resume.json',dict(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e),started=started,ended=time.time()))
            raise
if __name__=='__main__': main()
