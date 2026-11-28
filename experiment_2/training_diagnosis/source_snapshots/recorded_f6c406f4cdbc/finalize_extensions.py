"""Bounded actual completion, full-pool comparisons and fresh deployment measurements."""
from pathlib import Path
import json,os,time,subprocess,datetime,statistics,csv
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0');status=R/'results/extension_finalization.json'
def save(state,**extra):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=state,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(status)
def run(script,*args,log):
    cmd=[str(py),str(R/'scripts'/script),*map(str,args)];save('RUNNING',command=cmd,log=log)
    with (R/'logs/extension_finalization_commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,time=datetime.datetime.now().astimezone().isoformat()))+'\n')
    with (R/'logs'/log).open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    deadline=time.monotonic()+24*3600
    while True:
        upstream=[]
        for lane in ('gpu0','gpu1'):
            p=R/'results'/f'extension_queue_{lane}.json';upstream.append(json.loads(p.read_text()) if p.exists() else {})
        if any(s.get('status')=='FAILED' for s in upstream):raise RuntimeError('extension lane failed; inspect lane logs')
        if all(s.get('status')=='TRAINING_AND_FULL_POOL_COMPLETE' for s in upstream):break
        if time.monotonic()>deadline:raise TimeoutError('extension training not complete within24h')
        save('WAITING_FOR_TRAINING_AND_EVALUATION',upstream=upstream);time.sleep(30)
    original=json.loads((R/'results/development_pool/rows.json').read_text());extra=[]
    for lane,n in [('gpu0',72),('gpu1',18)]:
        rows=json.loads((R/'results/extension_evaluation'/lane/'rows.json').read_text());assert len(rows)==n;extra+=rows
    merged=original+extra;assert len({(r['method'],r['video']) for r in merged})==len(merged)==774
    out=R/'results/extension_comparison';out.mkdir(exist_ok=True);(out/'rows.json').write_text(json.dumps(merged,indent=2,allow_nan=False))
    run('summarize_pool.py','--input-dir',out,'--figure-prefix','extension_rd',log='extension-summary.log')
    (R/'logs/extension_hardware_before_profile.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    for folder,mode in [('P2_long10000','P2'),('P3_long1000','P3'),('D42_temporal5000','TP2')]:
        run('gpu_profile.py','--method',mode,'--checkpoint',R/'checkpoints/extensions'/folder/(mode+'.pt'),'--output-name',folder,log=f'extension-profile-{folder}.log')
    summary=json.loads((out/'summary.json').read_text());coverage={r['method']:r for r in summary['coverage']}
    lines=['# Extended temporal codec training results','',f"Generated {datetime.datetime.now().astimezone().isoformat()}; execution complete, figure/text review PENDING.",'','All18 historical development videos; independent generalization NOT VERIFIED. Sealed test NOT OPENED. Original DiT/VAE frozen. Previous684-row experiment preserved;90 new actual-stream measurements appended in a separate comparison directory.','', '| Method | Videos | Mean Bytes | Mean PSNR-Y | Uniform matched-rate delta/dB | Comparison videos |','|---|---:|---:|---:|---:|---:|']
    selected={'P2','P3','D42_f1_seed20261127_20000'}|{r['method'] for r in extra}
    for r in summary['summary']:
        if r['method'] in selected:
            c=coverage[r['method']];delta=c['mean_matched_rate_delta_db'];text=f'{delta:.4f}' if isinstance(delta,(int,float)) else str(delta)
            lines.append(f"| {r['method']} | {r['videos']} | {r['mean_actual_bytes']:.1f} | {r['mean_psnr_y']:.4f} | {text} | {c['matched_rate_videos']} |")
    controls=[]
    for video in sorted({r['video'] for r in extra}):
        selected_rows={r['method']:r for r in extra if r['video']==video};b=selected_rows['D42_temporal5000_TP0'];t=selected_rows['D42_temporal5000_TP2'];z=selected_rows['D42_temporal5000_TP2Z']
        assert b['psnr_y']==t['psnr_y']==z['psnr_y'] and b['latent_mse']==t['latent_mse']==z['latent_mse']
        controls.append(dict(video=video,TP2_vs_TP0_saving=1-t['actual_bytes']/b['actual_bytes'],history_vs_zero_saving=1-t['actual_bytes']/z['actual_bytes']))
    with (out/'D42_fixed_symbol_controls.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(controls[0]));writer.writeheader();writer.writerows(controls)
    lines+=['',f"D42 frozen-symbol TP2 mean byte saving versus normalized TP0: {statistics.mean(r['TP2_vs_TP0_saving'] for r in controls):.4%}; independent history contribution versus same-capacity zero-history head: {statistics.mean(r['history_vs_zero_saving'] for r in controls):.4%}. All18 reconstructions and canonical symbols exactly match their new D42 reference.",'','P2/P3 continuations retain optimizer, scheduler, RNG and full-video sampling position. D42 temporal heads use a different frozen base and training budget from original D16/10000 temporal heads, so that cross-family difference is not a single-factor history ablation.','', 'Matched-rate and matched-quality comparisons interpolate log(Bytes) only within each measured Pareto support; coverage can differ. Bytes and PSNR changes together do not establish equal-rate success. H.264 comparison CSVs retain unfavorable cases. LPIPS/DISTS, BD-rate, P4, DiT joint training and independent validation NOT RUN.','', 'Fresh GPU-start deployment timing (one warmup +five formal) is under results/performance/{P2_long10000,P3_long1000,D42_temporal5000}. Codec encoder never invokes VAE; both receiver endpoints are rawYUV420. Encoder speedup is not a two-end speedup.']
    (R/'extension_report.md').write_text('\n'.join(lines)+'\n');save('EXECUTION_COMPLETE_REVIEW_PENDING',original_rows=684,new_rows=90,total_rows=774,report=str(R/'extension_report.md'))
except Exception as e:save('FAILED',error=repr(e));raise
