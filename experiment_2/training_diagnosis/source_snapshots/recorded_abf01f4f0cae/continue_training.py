"""Authorized bounded training queues; tmux survives SSH/quota interruptions.
No job is marked COMPLETE before the command and real-stream evaluation succeed.
"""
import argparse,json,os,subprocess,time,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['gpu0','gpu1'],required=True);a=p.parse_args();R=Path(__file__).resolve().parents[1];REPO=R.parents[1];PY=REPO/'.venv/bin/python';env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0' if a.lane=='gpu0' else '1';state=R/'results'/f'queue_{a.lane}.json'
def save(status,**extra):
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,lane=a.lane,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(state)
def run(script,*args,log):
    command=[str(PY),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=command,log=log)
    with (R/'logs'/f'queue_{a.lane}_commands.jsonl').open('a') as journal:journal.write(json.dumps(dict(command=command,log=log,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def wait_for(paths):
    deadline=time.monotonic()+24*3600
    def ready(path):
        if not path.exists():return False
        try:value=json.loads(path.read_text())
        except (OSError,ValueError):return False
        return len(value)>=2 if isinstance(value,list) else True
    while not all(ready(p) for p in paths):
        if time.monotonic()>deadline:raise TimeoutError('prerequisite not ready within24h')
        save('WAITING_FOR_PREREQUISITES',paths=[str(p) for p in paths]);time.sleep(20)
try:
    wait_for([R/'results/E2_1e'/f'{mode}_D16/development_step1000.json' for mode in ('V1','V2')])
    for mode in ('V1','V2'):
        assert json.loads((R/'results/E2_1e'/f'{mode}_D16/vae_integrity.json').read_text())['unchanged']
    if a.lane=='gpu0':
        run('temporal_pixel.py','--steps',100,log='queued-P3-train.log')
        run('evaluate_joint.py','--method','P3',log='queued-P3-eval.log')
        run('pixel_train.py','--mode','V2','--steps',3000,log='queued-V2-mid3000.log')
        run('evaluate_development.py','--name','V2_D16','--steps',3000,'--stage','E2_1e',log='queued-V2-mid3000-eval.log')
        save('TRAINING_AND_EVALUATION_COMPLETE',checkpoint_steps=3000)
    else:
        for factor,name in [(.1,'V2_low_D16'),(10,'V2_high_D16')]:
            run('pixel_train.py','--mode','V2','--factor',factor,'--steps',1000,log=f'queued-{name}-1000.log')
            run('evaluate_development.py','--name',name,'--steps',1000,'--stage','E2_1e',log=f'queued-{name}-eval1000.log')
        wait_for([R/'results/E2_1e/V2_D16/development_step3000.json',R/'results/E2_2/P3/development.json'])
        run('extra_quality.py',log='queued-extra-quality.log')
        run('evaluate_pool.py','--include-p3',log='queued-full-pool.log')
        save('TRAINING_AND_FULL_POOL_COMPLETE')
except Exception as e:save('FAILED',error=repr(e));raise
