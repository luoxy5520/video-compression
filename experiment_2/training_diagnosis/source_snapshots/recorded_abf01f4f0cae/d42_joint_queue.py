"""Train next legal42-video joint phase after measured fixed-symbol work completes.
GPU0 lock prevents overlap with our formal deployment measurements.
"""
from pathlib import Path
import json,os,subprocess,time,datetime,fcntl
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');status=R/'results/d42_joint_queue.json'
def save(state,**extra):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=state,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(status)
def wait(path,target):
    deadline=time.monotonic()+24*3600
    while True:
        s=json.loads(path.read_text()) if path.exists() else {}
        if s.get('status')==target:return
        if s.get('status')=='FAILED':raise RuntimeError('prerequisite failed: '+str(path))
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        save('WAITING_FOR_PREREQUISITE',path=str(path),target=target);time.sleep(30)
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs/d42_joint_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    wait(R/'results/extension_queue_gpu0.json','TRAINING_AND_FULL_POOL_COMPLETE')
    # New joint phase: optimizer intentionally initialized afresh from frozen42-video codec.
    with (R/'logs/gpu0_exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('temporal_joint.py','--steps',10000,'--run-name','P2_D42_joint10000','--source-checkpoint',R/'checkpoints/extensions/D42_temporal5000/TP2.pt','--base-checkpoint',R/'checkpoints/D42_f1_seed20261127/step20000.pt',log='D42-joint10000-train.log')
        run('evaluate_extensions.py','--lane','gpu0','--run','P2_D42_joint10000:P2','--output-name','d42_joint',log='D42-joint10000-evaluation.log')
    wait(R/'results/extension_finalization.json','EXECUTION_COMPLETE_REVIEW_PENDING')
    out=R/'results/d42_joint_comparison';out.mkdir(exist_ok=True)
    previous=json.loads((R/'results/extension_comparison/rows.json').read_text());new=json.loads((R/'results/extension_evaluation/d42_joint/rows.json').read_text());assert len(previous)==774 and len(new)==18
    rows=previous+new;assert len({(r['method'],r['video']) for r in rows})==792
    (out/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    run('summarize_pool.py','--input-dir',out,'--figure-prefix','d42_joint_rd',log='D42-joint-comparison.log')
    run('gpu_profile.py','--method','P2','--checkpoint',R/'checkpoints/extensions/P2_D42_joint10000/P2.pt','--output-name','P2_D42_joint10000',log='D42-joint-deployment.log')
    result=next(r for r in json.loads((out/'summary.json').read_text())['summary'] if r['method']=='P2_D42_joint10000_P2')
    (R/'D42_joint_report.md').write_text('# D42 joint RD continuation\n\n'+json.dumps(result,indent=2)+'\n\nNew joint phase from frozenD42/20000 plusTP2/5000,10000 joint updates,42 legal training videos. Optimizer initialized anew, not an exact continuation of D16 or D42 latent-only optimizers. All18 historical development videos actually encoded/decoded and native-VAE evaluated. Full baseline comparison at results/d42_joint_comparison; GPU-start timing at results/performance/P2_D42_joint10000. Figure/text review PENDING. Independent validation and sealed test NOT RUN. Original DiT/VAE frozen.\n')
    save('EXECUTION_COMPLETE_REVIEW_PENDING',new_rows=18,total_rows=792,report=str(R/'D42_joint_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
