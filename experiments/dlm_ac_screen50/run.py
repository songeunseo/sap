"""CPU-only controller, progress, report, and explicit tmux launch."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import tempfile
from experiments.dlm_multiscale_ac50.artifacts import lock,read,write,freeze,sha,digest
from .prepare import ROOT,MULTI,LEGACY,REPO
from .core import ARMS,CONTRASTS
from .runtime_integrity import validate_job_receipt,write_job_receipt
SESSION='ac_screen50'

def _proc_info(pid):
    """Read Linux process identity, including its dedicated process group."""
    try:
        text=Path(f'/proc/{int(pid)}/stat').read_text()
        rest=text.split(') ',1)[1].split()
        return dict(state=rest[0],ppid=int(rest[1]),pgid=int(rest[2]),session=int(rest[3]),start_ticks=rest[19])
    except (FileNotFoundError,ProcessLookupError,ValueError,IndexError):
        return None

def proc_identity(pid):
    info=_proc_info(pid)
    return info['start_ticks'] if info else None

def _group_members(pgid):
    members=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            pid=int(path.parent.name);info=_proc_info(pid)
            if info and info['state']!='Z' and info['pgid']==int(pgid):members.append((pid,info))
        except (FileNotFoundError,ProcessLookupError,ValueError):
            continue
    return members

def _owned_group(row):
    """Return members of a group whose leader identity we recorded."""
    pid=int(row['pid']);expected=row.get('start_ticks');pgid=int(row.get('pgid',pid));session=row.get('session')
    leader=_proc_info(pid)
    if leader:
        if expected is not None and leader['start_ticks']!=expected:return []
        if leader['pgid']!=pgid:return []
        if session is not None and leader['session']!=int(session):return []
        return _group_members(pgid)
    # Once the leader is gone, require the persisted private session identity;
    # otherwise a recycled PID/group must never be signalled.
    members=_group_members(pgid)
    if not members or pgid!=pid or session is None:return []
    return [item for item in members if item[1]['session']==int(session)]

def job_graph():
    jobs=[dict(id='smoke',kind='smoke',family='Square',deps=[])]
    for arm in ['Short','Multi','MS-A','Path','All']:
        jobs.append(dict(id=arm,kind='legacy',family='Multi',method='A' if arm=='MS-A' else arm,deps=[]))
    for family in ['Square','Vector']:
        for split in ['calibration','diagnostic']:jobs.append(dict(id=f'{family}_teacher_{split}',family=family,kind='teacher',split=split,deps=['smoke']))
        jobs.append(dict(id=f'{family}_uniform',family=family,kind='uniform',deps=[f'{family}_teacher_calibration']))
        for b in range(0,32,4):jobs.append(dict(id=f'{family}_probe_{b:02d}',family=family,kind='probe',blocks=list(range(b,b+4)),deps=[f'{family}_teacher_calibration',f'{family}_uniform']))
        jobs.append(dict(id=f'{family}_allocation',family=family,kind='allocation',deps=[f'{family}_probe_{b:02d}' for b in range(0,32,4)]))
        for obj in ['A','AC']:jobs.append(dict(id=f'{family}-{obj}',family=family,kind='candidate',objective=obj,deps=[f'{family}_allocation',f'{family}_teacher_diagnostic']))
    jobs.append(dict(id='Exchange-AC',family='Exchange',kind='exchange',deps=['Vector_teacher_diagnostic']))
    return jobs

def select_ready(jobs,done,active,last_family=None):
    ready=[j for j in jobs if j['id'] not in done|active and set(j['deps'])<=done]
    smoke=next((j for j in ready if j['kind']=='smoke'),None)
    if smoke is not None:return smoke
    families=['Multi','Square','Vector','Exchange']
    if last_family in families:
        p=families.index(last_family)+1;families=families[p:]+families[:p]
    for f in families:
        for j in ready:
            if j['family']==f:return j
    return None

def rows_for(arm):
    from experiments.dlm_multiscale_ac50.evaluation import read_predictions
    req=read(MULTI/'requests.json')['development'];c=read(MULTI/'config.json')
    if arm.startswith('legacy_'):return read(MULTI/'cached_development.json')[arm[7:]]
    old=arm in ['MS-A','Short','Path','All','Multi'];name='A' if arm=='MS-A' else arm
    folder=(MULTI/'gsm8k/development' if old else ROOT/'gsm8k')/name
    if not (folder/'identity.json').exists():return []
    fp=read(folder/'identity.json')['fingerprint']
    if fp['config_sha256']!=sha((MULTI/'config.json') if old else ROOT/'manifest.json'):raise ValueError('Result config differs')
    mi=read((MULTI if old else ROOT)/'candidates'/name/'model_identity.json')
    if fp['mask_identity']!=mi['mask_identity'] or fp['sparse_model_sha256']!=mi['sparse_model_sha256'] or fp['requests_sha256']!=sha(MULTI/'requests.json'):raise ValueError('Model/request identity differs')
    return read_predictions(folder,req,fp,c['protocol_hash'],allow_partial=True)

def status(as_json=False):
    state=read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else dict(status='not_launched')
    data={a:dict(total=len(rr:=rows_for(a)),correct=sum(r['correct'] for r in rr)) for a in [*ARMS,'legacy_uniform','legacy_A','legacy_AC']}
    progress={}
    execution_workers={str(row.get('job',{}).get('id')):row for row in state.get('workers',[]) if isinstance(row,dict)}
    for progress_root in [MULTI,ROOT]:
        for p in (progress_root/'progress').glob('*.json'):
            r=read(p);worker=str(r.get('worker',''))
            if not r.get('gpu') or r.get('stage') in ['complete','failed']:continue
            owner=execution_workers.get(worker)
            if owner is not None:
                alive=proc_identity(r.get('pid'))==owner.get('start_ticks') and int(owner.get('pid'))==int(r.get('pid'))
            else:
                cmd=_controller_command(r.get('pid'))
                alive=bool(proc_identity(r.get('pid')) and 'experiments.dlm_' in cmd and ' worker ' in f' {cmd} ')
            if alive:progress[worker]=r
    result=dict(execution=state,scores=data,active=progress,legacy_stage=read(ROOT/'legacy_resume.json') if (ROOT/'legacy_resume.json').exists() else None)
    if as_json:print(json.dumps(result,indent=2));return result
    print('AC 50% mini100 | '+time.strftime('%Y-%m-%d %H:%M:%S %Z'));print('Screen:',state['status'])
    for a,r in data.items():print(f'{a:18} {r["correct"]:3}/{r["total"]:<3} '+('complete' if r['total']==100 else 'partial' if r['total'] else 'pending'))
    for key,r in progress.items():
        eta=r.get('eta_seconds');eta=f'{eta/60:.1f} min' if eta is not None else 'unknown'
        print(f'{key}: {r["stage"]} {r["completed"]}/{r["total"]}, GPU {r["gpu"]}, PID {r["pid"]}, ETA {eta}')
    print('Logs:',ROOT/'logs');return result

def _atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent),text=True)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        with contextlib.suppress(FileNotFoundError):os.unlink(tmp)

def report():
    from .reporting import build_report
    rows={a:rows_for(a) for a in [*ARMS,'legacy_uniform','legacy_A','legacy_AC']}
    result,markdown=build_report(ROOT,MULTI,rows)
    write(ROOT/'report.json',result)
    _atomic_text(ROOT/'report.md',markdown)
    print(json.dumps(result.get('scores',{}),indent=2));return result

def command(*args):return [sys.executable,'-B','-u','-m','experiments.dlm_ac_screen50.run',*args]

def terminate_owned(items,grace=10):
    """Terminate only the private process groups recorded for these workers."""
    rows=[];seen=set()
    for row in items:
        if not row or 'pid' not in row:continue
        pgid=int(row.get('pgid',row['pid']))
        if pgid in seen:continue
        seen.add(pgid);rows.append(row)
    for row in rows:
        if _owned_group(row):
            with contextlib.suppress(ProcessLookupError,PermissionError):
                os.killpg(int(row.get('pgid',row['pid'])),signal.SIGTERM)
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline:
        if not any(_owned_group(row) for row in rows):break
        time.sleep(.2)
    for row in rows:
        if _owned_group(row):
            with contextlib.suppress(ProcessLookupError,PermissionError):
                os.killpg(int(row.get('pgid',row['pid'])),signal.SIGKILL)
    for row in rows:
        proc=row.get('proc')
        if proc is not None:
            with contextlib.suppress(Exception):proc.wait(timeout=1)

def _worker_state(row):
    return {k:row[k] for k in ('pid','start_ticks','pgid','session','gpu','job','log','started') if k in row}

def pipeline(gpus,after_legacy=False):
    if not os.environ.get('TMUX'):raise RuntimeError('Requires tmux')
    from .prepare import validate
    manifest=validate();jobs=job_graph();jobs_path=ROOT/'jobs.json';freeze(jobs_path,jobs)
    jobs_sha256=sha(jobs_path);started=time.time();active=[];done=set();last=None
    def interrupted(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    base=dict(pid=os.getpid(),start_ticks=proc_identity(os.getpid()),gpus=gpus,started=started,config_sha256=sha(ROOT/'manifest.json'))
    ownership=False
    try:
        # The optional flag is retained for command compatibility.  Legacy
        # receipts are reusable DAG nodes; a global wait is unsafe because it
        # starves independent families and allows a race with the bootstrap.
        from experiments.dlm_multiscale_ac50.run import idle_devices
        idle_devices(gpus)
        with lock(ROOT/'screen.lock'),lock(MULTI/'pipeline.lock'):
            ownership=True
            for j in jobs:
                receipt=ROOT/'jobs_done'/f'{j["id"]}.json'
                if receipt.exists():
                    validate_job_receipt(receipt,ROOT,MULTI,j,base['config_sha256'],jobs_sha256)
                    done.add(j['id'])
                elif j['kind']=='legacy' and len(rows_for(j['id']))==100:
                    write_job_receipt(receipt,ROOT,MULTI,j,base['config_sha256'],jobs_sha256)
                    done.add(j['id'])
            while len(done)<len(jobs):
                for gpu in gpus:
                    if gpu in {r['gpu'] for r in active}:continue
                    j=select_ready(jobs,done,{r['job']['id'] for r in active},last)
                    if not j:continue
                    last=j['family']
                    if j['kind']=='allocation':
                        from .allocation import allocate
                        allocate(j['family'])
                        write_job_receipt(ROOT/'jobs_done'/f'{j["id"]}.json',ROOT,MULTI,j,base['config_sha256'],jobs_sha256)
                        done.add(j['id']);continue
                    argv=command('worker','--job',j['id'])
                    if j['kind']=='legacy':argv=[sys.executable,'-B','-u','-m','experiments.dlm_multiscale_ac50.run','--root',str(MULTI),'worker','--job','candidate','--method',j['method'],'--split','development']
                    logfile=ROOT/'logs'/f'{j["id"]}.log';logfile.parent.mkdir(parents=True,exist_ok=True);stream=logfile.open('a')
                    p=subprocess.Popen(argv,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    info=_proc_info(p.pid) or {}
                    active.append(dict(proc=p,pid=p.pid,start_ticks=info.get('start_ticks',proc_identity(p.pid)),pgid=info.get('pgid',p.pid),session=info.get('session'),gpu=gpu,job=j,stream=stream,log=str(logfile),started=time.time()))
                    # Persist ownership immediately after spawn so stop/recovery
                    # can reclaim a worker even before its first progress tick.
                    write(ROOT/'execution.json',dict(base,status='running',completed_jobs=sorted(done),workers=[_worker_state(r) for r in active]))
                for r in list(active):
                    code=r['proc'].poll()
                    if code is None:continue
                    if code:
                        raise RuntimeError(f'{r["job"]["id"]} exit {code}; {r["log"]}')
                    # Keep the record in active until receipt validation passes;
                    # a malformed/missing output must trigger group cleanup.
                    write_job_receipt(ROOT/'jobs_done'/f'{r["job"]["id"]}.json',ROOT,MULTI,r['job'],base['config_sha256'],jobs_sha256)
                    r['stream'].close();active.remove(r);done.add(r['job']['id'])
                write(ROOT/'execution.json',dict(base,status='running',completed_jobs=sorted(done),workers=[_worker_state(r) for r in active]))
                if not active and not select_ready(jobs,done,set()):
                    if len(done)<len(jobs):raise RuntimeError('Blocked dependency graph')
                time.sleep(1)
            result=report();write(ROOT/'execution.json',dict(base,status=result['status'],completed_jobs=sorted(done),workers=[],ended=time.time(),obsidian_sync='pending'))
    except BaseException as e:
        terminate_owned(active)
        for row in active:
            with contextlib.suppress(Exception):row.get('stream').close()
        if ownership:
            write(ROOT/'execution.json',dict(base,status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e),completed_jobs=sorted(done),workers=[],ended=time.time(),obsidian_sync='pending'))
        raise

def launch(gpus,dry=False,after=False):
    if not gpus or len(set(gpus))!=len(gpus) or any(not x.isdigit() for x in gpus):raise ValueError('Explicit unique GPU indices required')
    from .prepare import validate
    m=validate();jobs=job_graph()
    if dry:
        print(json.dumps(dict(dry_run=True,gpu_queried=False,gpu_used=False,jobs=jobs,preparation=read(ROOT/'preparation_receipt.json'),costs=m['costs'],smoke_cap=32),indent=2));return
    # Serialize the check/reservation/tmux creation window.  The pipeline then
    # takes the same screen/pipeline locks after this lock is released.
    with lock(ROOT/'launch.lock'):
        if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('Screen tmux already exists')
        from experiments.dlm_multiscale_ac50.run import idle_devices
        idle_devices(gpus)
        with lock(ROOT/'screen.lock'),lock(MULTI/'pipeline.lock'):
            argv=command('pipeline','--gpus',','.join(gpus),*(['--after-legacy'] if after else []))
            env={k:os.environ[k] for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TOKENIZERS_PARALLELISM'] if k in os.environ}
            wrapped=['env','CUDA_VISIBLE_DEVICES=',*[f'{k}={v}' for k,v in env.items()],*argv]
            log=ROOT/'logs/controller.log';log.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(REPO),shlex.join(wrapped)+' >> '+shlex.quote(str(log))+' 2>&1'],check=True)
    print(json.dumps(dict(launched=True,session=SESSION,gpus=gpus,after_legacy=after)))

def _controller_command(pid):
    try:return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (FileNotFoundError,ProcessLookupError,TypeError,ValueError):return ''

def _owned_controller_command(cmd):
    return any(module in cmd for module in ('experiments.dlm_ac_screen50.run','experiments.dlm_ac_screen50.resume_legacy','experiments.dlm_multiscale_ac50.run'))

def _active_status(value):
    return value in {'running','waiting_for_legacy','stopping'}

def stop():
    states=[ROOT/'execution.json',ROOT/'legacy_resume.json']
    controllers=[];worker_rows=[]
    for path in states:
        if not path.exists():continue
        row=read(path)
        if not _active_status(row.get('status')):continue
        pid=row.get('pid')
        if pid and proc_identity(pid):
            cmd=_controller_command(pid)
            if not _owned_controller_command(cmd):
                raise RuntimeError('Refusing unrelated PID')
            expected=row.get('start_ticks')
            if expected and proc_identity(pid)!=expected:raise RuntimeError('PID reused')
            controllers.append((pid,expected))
        if path.name=='execution.json':worker_rows.extend(row.get('workers',[]))
    for pid,expected in controllers:
        if proc_identity(pid)==expected if expected else proc_identity(pid):
            with contextlib.suppress(ProcessLookupError):os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        live=False
        for path in states:
            if path.exists() and _active_status(read(path).get('status')):live=True
        controllers_live=any(proc_identity(pid)==expected if expected else proc_identity(pid) for pid,expected in controllers)
        if not live and not controllers_live:break
        time.sleep(.2)
    # A crashed controller cannot perform its own finally block.  Recover its
    # persisted worker groups only after the bounded controller wait.
    if controllers and any(proc_identity(pid)==expected if expected else proc_identity(pid) for pid,expected in controllers):
        for pid,expected in controllers:
            if (proc_identity(pid)==expected if expected else proc_identity(pid)):
                with contextlib.suppress(ProcessLookupError):os.kill(pid,signal.SIGKILL)
    terminate_owned(worker_rows,grace=5)
    for path in states:
        if not path.exists():continue
        row=read(path)
        if _active_status(row.get('status')):
            row['status']='interrupted';row['stopped_at']=time.time();row['workers']=[]
            write(path,row)
    print('Stopped owned controllers; checkpoints preserved.')

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    for x in ['prepare','validate','report','stop']:s.add_parser(x)
    a=s.add_parser('status');a.add_argument('--json',action='store_true')
    for x in ['launch','pipeline']:
        a=s.add_parser(x);a.add_argument('--gpus',required=True);a.add_argument('--after-legacy',action='store_true');a.add_argument('--dry-run',action='store_true')
    a=s.add_parser('worker');a.add_argument('--job',required=True)
    args=p.parse_args()
    if args.cmd!='worker':os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.cmd=='prepare':
        from .prepare import prepare
        prepare();print('Prepared; no model/GPU work.')
    elif args.cmd=='validate':
        from .prepare import validate
        validate();print('Validated frozen inputs.')
    elif args.cmd=='status':status(args.json)
    elif args.cmd=='report':report()
    elif args.cmd=='stop':stop()
    elif args.cmd=='launch':launch(args.gpus.split(','),args.dry_run,args.after_legacy)
    elif args.cmd=='pipeline':pipeline(args.gpus.split(','),args.after_legacy)
    else:
        from .worker import main as worker
        job=next(j for j in job_graph() if j['id']==args.job)
        with lock(ROOT/'locks'/f'{args.job}.lock'):worker(job)
if __name__=='__main__':main()
