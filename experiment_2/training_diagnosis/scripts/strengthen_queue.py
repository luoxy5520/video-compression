"""Continue measured positive VAE-aware result and fill matched-budget endpoint points."""
from pathlib import Path
import argparse,os,json,subprocess,time,datetime,fcntl
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['gpu0','gpu1'],required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if a.lane=='gpu0' else '1');status=R/'results'/f'strengthen_{a.lane}.json'
def save(state,**extra):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=state,lane=a.lane,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(status)
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs'/f'strengthen_{a.lane}_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    deadline=time.monotonic()+24*3600
    while True:
        path=R/'results/index_lookup_profile_status.json';s=json.loads(path.read_text()) if path.exists() else {}
        if s.get('status')=='EXECUTION_COMPLETE_REVIEW_PENDING':break
        if s.get('status')=='FAILED':raise RuntimeError('index profiling failed; inspect logs')
        if time.monotonic()>deadline:raise TimeoutError('index profiling prerequisite')
        save('WAITING_FOR_PROFILING');time.sleep(20)
    # Shared locks permit the two training GPUs concurrently; formal profiles use exclusive lock.
    with (R/'logs/gpu0_exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        if a.lane=='gpu1':
            run('temporal_pixel.py','--steps',3000,'--run-name','P3_D42_3000','--source-checkpoint',R/'checkpoints/extensions/P2_D42_joint10000/P2.pt','--initial-state',R/'checkpoints/extensions/P3_D42_1000/latest.pt',log='P3-D42-3000-train.log')
            for basename,name in [('step2000','P3_D42_step2000'),('P3','P3_D42_step3000')]:
                run('evaluate_extensions.py','--lane','gpu1','--run','P3_D42_3000:'+basename,'--output-name',name,log=name+'-evaluate.log')
        else:
            for level in ('low','high'):
                name='P3_D42_'+level+'1000'
                run('temporal_pixel.py','--steps',1000,'--run-name',name,'--source-checkpoint',R/'checkpoints/extensions'/('P2_D42_'+level+'10000')/'P2.pt',log=name+'-train.log')
                run('evaluate_extensions.py','--lane','gpu0','--run',name+':P3','--output-name',name,log=name+'-evaluate.log')
    save('TRAINING_AND_FULL_POOL_COMPLETE')
except Exception as e:save('FAILED',error=repr(e));raise
