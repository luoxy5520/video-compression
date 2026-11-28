"""Actual full-pool and deployment completion for four new VAE-aware checkpoints."""
from pathlib import Path
import os,json,time,subprocess,datetime
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');state=R/'results/strengthen_finalization.json'
def save(status,**extra):
    p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));p.replace(state)
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    deadline=time.monotonic()+24*3600
    while True:
        states=[]
        for lane in ('gpu0','gpu1'):
            p=R/'results'/f'strengthen_{lane}.json';states.append(json.loads(p.read_text()) if p.exists() else {})
        if any(s.get('status')=='FAILED' for s in states):raise RuntimeError('upstream strengthen lane failed')
        if all(s.get('status')=='TRAINING_AND_FULL_POOL_COMPLETE' for s in states):break
        if time.monotonic()>deadline:raise TimeoutError('strengthen lanes')
        save('WAITING_FOR_TRAINING_AND_EVALUATION',upstream=states);time.sleep(30)
    runs=[('P3_D42_3000','step2000','P3_D42_step2000'),('P3_D42_3000','P3','P3_D42_step3000'),('P3_D42_low1000','P3','P3_D42_low1000'),('P3_D42_high1000','P3','P3_D42_high1000')]
    rows=json.loads((R/'results/d42_pixel_comparison/rows.json').read_text());assert len(rows)==846
    for folder,base,name in runs:
        new=json.loads((R/'results/extension_evaluation'/name/'rows.json').read_text());assert len(new)==18;rows+=new
    assert len(rows)==918 and len({(r['method'],r['video']) for r in rows})==918
    out=R/'results/strengthen_comparison';out.mkdir(exist_ok=True);(out/'rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    run('summarize_pool.py','--input-dir',out,'--figure-prefix','strengthen_rd',log='strengthen-summary.log')
    (R/'logs/strengthen_hardware_before_profiles.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    for folder,base,name in runs:
        run('gpu_profile.py','--method','P3','--checkpoint',R/'checkpoints/extensions'/folder/(base+'.pt'),'--output-name',name,log=name+'-profile.log')
    summary=json.loads((out/'summary.json').read_text());names={r['method'] for r in rows if r['method'].startswith('P3_D42')};selected=[r for r in summary['summary'] if r['method'] in names]
    (R/'strengthen_report.md').write_text('# Stronger frozen-VAE-aware codec training\n\n'+json.dumps(selected,indent=2)+'\n\n918 method-video evaluations across18 historical development videos. New endpoints low/high each1000 steps from corresponding matched-budget P2 models;compare to existing middle1000. Middle2000/3000 are separate exact-state continuation checkpoints, not equal-budget endpoint points. Full actual-stream, source/input hashes, SSIM and measured-support Uniform/H264 comparisons retained at results/strengthen_comparison. Source DiT/VAE weights frozen;sealed test and independent validation NOT RUN. Formal timing1warmup+5formal follows all training under exclusive experiment lock. Figure/text review PENDING.\n')
    save('EXECUTION_COMPLETE_REVIEW_PENDING',new_rows=72,total_rows=918,report=str(R/'strengthen_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
