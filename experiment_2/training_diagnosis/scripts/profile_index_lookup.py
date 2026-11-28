"""Fresh paired full-pipeline timings after real-stream equivalence passes."""
from pathlib import Path
import os,json,subprocess,statistics,datetime,hashlib
R=Path(__file__).resolve().parents[1];repo=R.parents[1];py=repo/'.venv/bin/python';state=R/'results/index_lookup_profile_status.json';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
def save(status,**extra):
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,updated=datetime.datetime.now().astimezone().isoformat(),**extra),indent=2));tmp.replace(state)
try:
    audit=json.loads((R/'results/index_lookup_audit/status.json').read_text());assert audit['status']=='PASS' and audit['historical_videos']==18
    ck=R/'checkpoints/extensions/P2_D42_joint10000/P2.pt'
    assert audit['source_sha256']==hashlib.sha256((R/'scripts/index_lookup.py').read_bytes()).hexdigest()
    assert audit['checkpoint_sha256']==hashlib.sha256(ck.read_bytes()).hexdigest()
    for name,flags in [('index_standard',[]),('index_bucketize',['--bucketize-indexes'])]:
        cmd=[str(py),str(R/'scripts/gpu_profile.py'),'--method','P2','--checkpoint',str(ck),'--output-name',name,*flags];save('RUNNING',command=cmd)
        with (R/'logs'/f'{name}-profile.log').open('a') as f:subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    base=R/'results/performance/index_standard';fast=R/'results/performance/index_bucketize'
    aa=json.loads((base/'deployment_benchmark.json').read_text());bb=json.loads((fast/'deployment_benchmark.json').read_text());assert len(aa)==len(bb)==2
    rows=[]
    for idx,(a,b) in enumerate(zip(aa,bb,strict=True)):
        assert a['video']==b['video'] and a['actual_bytes']==b['actual_bytes'] and a['quality']==b['quality']
        assert (base/f'gpu_dev{idx}.bin').read_bytes()==(fast/f'gpu_dev{idx}.bin').read_bytes()
        enc=lambda r:statistics.mean(s['full_gpu_latent_to_bitstream_seconds'] for s in r['codec_samples'])
        ends=lambda r:statistics.mean(s['post_dit_two_ends_seconds'] for s in r['codec_samples'])
        rows.append(dict(video=a['video'],actual_bytes=a['actual_bytes'],psnr_y=a['quality']['psnr_y'],bitstream_identical=True,baseline_encoder_seconds=enc(a),bucketize_encoder_seconds=enc(b),encoder_latency_saving_fraction=1-enc(b)/enc(a),baseline_two_ends_seconds=ends(a),bucketize_two_ends_seconds=ends(b),two_ends_saving_fraction=1-ends(b)/ends(a)))
    result=dict(rows=rows,scope='two representative historical videos;each method1warmup+5formal;all timing samples retained;no training active;encoder and both-end results separate;adverse timing preserved')
    (R/'results/index_lookup_performance_comparison.json').write_text(json.dumps(result,indent=2));save('EXECUTION_COMPLETE_REVIEW_PENDING',videos=2,all_bitstreams_identical=True)
except Exception as e:save('FAILED',error=repr(e));raise
