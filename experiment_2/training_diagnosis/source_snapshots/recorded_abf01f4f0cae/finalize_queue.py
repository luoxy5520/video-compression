"""Bounded completion queue for actual evaluation, profiling and measured reports."""
import os,json,time,subprocess,datetime,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];REPO=R.parents[1];PY=REPO/'.venv/bin/python';state=R/'results/finalization_queue.json';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
def save(status,**extra):
    t=state.with_suffix('.tmp');t.write_text(json.dumps(dict(status=status,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));t.replace(state)
def run(script,*args,log):
    cmd=[str(PY),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs/finalization_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    deadline=time.monotonic()+24*3600
    while True:
        states=[]
        for lane in ('gpu0','gpu1'):
            p=R/'results'/f'queue_{lane}.json';states.append(json.loads(p.read_text()) if p.exists() else {})
        if any(s.get('status')=='FAILED' for s in states):raise RuntimeError('upstream training/evaluation queue failed; inspect queue state')
        if states[0].get('status')=='TRAINING_AND_EVALUATION_COMPLETE' and states[1].get('status')=='TRAINING_AND_FULL_POOL_COMPLETE':break
        if time.monotonic()>deadline:raise TimeoutError('upstream queues not ready within24h')
        save('WAITING_FOR_TRAINING_AND_FULL_POOL',upstream=states);time.sleep(30)
    (R/'logs/hardware_before_final_profile.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    run('gpu_profile.py','--method','P2',log='final-profile-P2.log');run('gpu_profile.py','--method','P3',log='final-profile-P3.log')
    for script in ('summarize_diagnosis.py','summarize_scale.py','summarize_pixel.py','summarize_temporal.py','summarize_pool.py','build_report.py'):run(script,log='final-'+script+'.log')
    hashes=[]
    for folder in ('checkpoints','results'):
        for p in (R/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.pt','.lvc','.lvc2','.bin','.264'):
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                hashes.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    (R/'results/artifact_hashes.json').write_text(json.dumps(hashes,indent=2));save('EXECUTION_COMPLETE_REVIEW_PENDING',artifacts=len(hashes),report=str(R/'final_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
