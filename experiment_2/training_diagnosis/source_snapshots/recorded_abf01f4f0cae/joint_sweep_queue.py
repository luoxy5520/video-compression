"""Matched-source42-video joint RD sweep; bounded tasks, actual complete-pool evaluation."""
from pathlib import Path
import json,os,subprocess,time,datetime,fcntl
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');status=R/'results/joint_sweep_queue.json'
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
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs/joint_sweep_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    wait(R/'results/extension_evaluation/d42_joint/status.json','COMPLETE')
    with (R/'logs/gpu0_exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for factor,level in [(.1,'low'),(10,'high')]:
            name=f'P2_D42_{level}10000'
            run('temporal_joint.py','--steps',10000,'--run-name',name,'--source-checkpoint',R/'checkpoints/extensions/D42_temporal5000/TP2.pt','--base-checkpoint',R/'checkpoints/D42_f1_seed20261127/step20000.pt','--lambda-factor',factor,log=f'{name}-train.log')
            run('evaluate_extensions.py','--lane','gpu0','--run',name+':P2','--output-name',name,log=f'{name}-evaluate.log')
    wait(R/'results/d42_joint_queue.json','EXECUTION_COMPLETE_REVIEW_PENDING')
    previous=json.loads((R/'results/d42_joint_comparison/rows.json').read_text());new=[]
    for name in ('P2_D42_low10000','P2_D42_high10000'):
        rows=json.loads((R/'results/extension_evaluation'/name/'rows.json').read_text());assert len(rows)==18;new+=rows
    rows=previous+new;assert len(rows)==828 and len({(r['method'],r['video']) for r in rows})==828
    out=R/'results/joint_sweep_comparison';out.mkdir(exist_ok=True);(out/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    run('summarize_pool.py','--input-dir',out,'--figure-prefix','joint_sweep_rd',log='joint-sweep-summary.log')
    for name in ('P2_D42_low10000','P2_D42_high10000'):
        run('gpu_profile.py','--method','P2','--checkpoint',R/'checkpoints/extensions'/name/'P2.pt','--output-name',name,log=name+'-profile.log')
    summary=json.loads((out/'summary.json').read_text());coverage={r['method']:r for r in summary['coverage']}
    lines=['# D42 joint RD: three matched-source working points','','Same frozen starting D42/20000 +TP2/5000 weights;42 legal training videos;seed20261127;10000 joint updates each;lambda factors0.1/1/10;new optimizers for all three phases. No original Wan parameters trained.','', '| Method | Videos | Mean Bytes | Mean PSNR-Y | Uniform matched-rate delta/dB | Matched videos |','|---|---:|---:|---:|---:|---:|']
    for r in summary['summary']:
        if r['method'].startswith('P2_D42'):
            c=coverage[r['method']];lines.append(f"| {r['method']} | {r['videos']} | {r['mean_actual_bytes']:.1f} | {r['mean_psnr_y']:.4f} | {c['mean_matched_rate_delta_db']} | {c['matched_rate_videos']} |")
    lines+=['','All three models independently stream-decoded on first video and actually stream-decoded/nativeVAE evaluated on all18 historical development videos. Symbols may change; these are lossy joint models, not fixed-symbol savings. Adverse or nonmonotonic points retained. Full per-video metrics, main/hyper/header rates, probability diagnostics and measured-support-only Uniform/H264 comparisons are under results/joint_sweep_comparison. No extrapolation or validated BD-rate. Independent validation and sealed test NOT RUN. Figure/text review PENDING. Formal GPU-start profiles use one warmup +five repetitions under the GPU0 experiment lock; encoder speedup is not total two-end speedup.']
    (R/'joint_sweep_report.md').write_text('\n'.join(lines)+'\n');save('EXECUTION_COMPLETE_REVIEW_PENDING',new_rows=36,total_rows=828,report=str(R/'joint_sweep_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
