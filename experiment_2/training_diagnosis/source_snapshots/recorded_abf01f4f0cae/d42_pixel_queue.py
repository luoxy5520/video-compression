"""VAE-aware training from measured stronger42-video joint codec, then all18 evaluation."""
from pathlib import Path
import json,os,subprocess,time,datetime,fcntl
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';status=R/'results/d42_pixel_queue.json'
def save(state,**extra):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=state,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(status)
def wait(path,target):
    deadline=time.monotonic()+24*3600
    while True:
        s=json.loads(path.read_text()) if path.exists() else {}
        if s.get('status')==target:return
        if s.get('status')=='FAILED':raise RuntimeError(str(path))
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        save('WAITING_FOR_PREREQUISITE',path=str(path));time.sleep(30)
def run(script,*args,log,device='1'):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log,physical_gpu=device)
    with (R/'logs/d42_pixel_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,device=device,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    wait(R/'results/extension_queue_gpu1.json','TRAINING_AND_FULL_POOL_COMPLETE')
    wait(R/'results/extension_evaluation/P2_D42_high10000/status.json','COMPLETE')
    # Hold the experiment profile lock while training on GPU1: no host/GPU timing overlap.
    with (R/'logs/gpu0_exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('temporal_pixel.py','--steps',1000,'--run-name','P3_D42_1000','--source-checkpoint',R/'checkpoints/extensions/P2_D42_joint10000/P2.pt',log='P3-D42-1000-train.log')
        run('evaluate_extensions.py','--lane','gpu1','--run','P3_D42_1000:P3','--output-name','d42_pixel',log='P3-D42-1000-evaluation.log')
    wait(R/'results/joint_sweep_queue.json','EXECUTION_COMPLETE_REVIEW_PENDING')
    old=json.loads((R/'results/joint_sweep_comparison/rows.json').read_text());new=json.loads((R/'results/extension_evaluation/d42_pixel/rows.json').read_text());assert len(old)==828 and len(new)==18
    rows=old+new;assert len({(r['method'],r['video']) for r in rows})==846
    out=R/'results/d42_pixel_comparison';out.mkdir(exist_ok=True);(out/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    run('summarize_pool.py','--input-dir',out,'--figure-prefix','d42_pixel_rd',log='P3-D42-summary.log',device='0')
    run('gpu_profile.py','--method','P3','--checkpoint',R/'checkpoints/extensions/P3_D42_1000/P3.pt','--output-name','P3_D42_1000',log='P3-D42-profile.log',device='0')
    summary=json.loads((out/'summary.json').read_text());measured=[r for r in summary['summary'] if r['method'] in ('P2_D42_joint10000_P2','P3_D42_1000_P3')]
    (R/'D42_pixel_report.md').write_text('# Frozen-VAE-aware42-video joint fine tuning\n\n'+json.dumps(measured,indent=2)+'\n\nP3_D42 starts from the actual measured P2_D42_joint10000, trains1000 full-sequence steps over the same42 legal train videos, with new optimizer and pixel-gradient calibration. Source codec and original DiT/VAE weights preserved. All18 historical development videos actually stream decoded and nativeVAE evaluated. Full measured-support Uniform/H264 comparisons at results/d42_pixel_comparison; fresh GPU-start profiling at results/performance/P3_D42_1000. No original model training or sealed test access. Independent validation and perceptual-model metrics NOT RUN. Figure/text review PENDING.\n')
    save('EXECUTION_COMPLETE_REVIEW_PENDING',new_rows=18,total_rows=846,report=str(R/'D42_pixel_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
