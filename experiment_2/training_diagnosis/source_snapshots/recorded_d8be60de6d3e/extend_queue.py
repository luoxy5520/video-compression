"""User-authorized stronger training, isolated from completed experiment artifacts."""
import argparse,subprocess,os,json,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['gpu0','gpu1'],required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if a.lane=='gpu0' else '1');state=R/'results'/f'extension_queue_{a.lane}.json'
def save(status,**extra):
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,updated=datetime.datetime.now().astimezone().isoformat(),lane=a.lane,**extra),indent=2));tmp.replace(state)
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs'/f'extension_commands_{a.lane}.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    if a.lane=='gpu0':
        run('temporal_joint.py','--steps',10000,'--run-name','P2_long10000','--initial-state',R/'checkpoints/temporal_joint/latest.pt',log='extension-P2-10000.log')
        run('temporal_run.py','--base-name','D42_f1_seed20261127','--base-step',20000,'--steps',5000,'--run-name','D42_temporal5000','--modes','TP0','TP2','TP2Z',log='extension-D42-temporal5000.log')
    else:
        run('temporal_pixel.py','--steps',1000,'--run-name','P3_long1000','--initial-state',R/'checkpoints/temporal_pixel/latest.pt',log='extension-P3-1000.log')
    run('evaluate_extensions.py','--lane',a.lane,log=f'extension-evaluate-{a.lane}.log')
    save('TRAINING_AND_FULL_POOL_COMPLETE')
except Exception as e:save('FAILED',error=repr(e));raise
