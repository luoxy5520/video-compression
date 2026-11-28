"""Real streams and frozen native VAE evaluation of isolated continuation runs."""
from pixels import *
from temporal_entropy_model import load_temporal
from rate_diagnostics import temporal_rates
from pytorch_msssim import ssim
import argparse
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['gpu0','gpu1'],required=True);p.add_argument('--run',help='isolated folder:checkpoint basename');p.add_argument('--output-name');a=p.parse_args()
names=[tuple(a.run.split(':'))] if a.run else [('P2_long10000','P2'),('D42_temporal5000','TP0'),('D42_temporal5000','TP2'),('D42_temporal5000','TP2Z')] if a.lane=='gpu0' else [('P3_long1000','P3')]
out=ROOT/'results/extension_evaluation'/(a.output_name or a.lane);out.mkdir(parents=True,exist_ok=True)
configs=[]
for folder,mode in names:
    ck=ROOT/'checkpoints/extensions'/folder/(mode+'.pt');m,c=load_temporal(ck)
    configs.append((folder+'_'+mode,m,c,ck,sha(ck)))
records=[r for r in json.loads((E2/'dataset_manifest.json').read_text())['records'] if r['role']=='historical_development' and r['exists']];assert len(records)==18
input_hashes={r['path']:r['file_sha256'] for r in json.loads((E2.parent/'experiment_1/results/input_manifest.json').read_text())}
for name,m,c,ck,digest in configs:assert not ({r['path'] for r in records}&set(c['train_paths']))
save(out/'source_snapshot.json',dict(models={n:d for n,m,c,k,d in configs},scripts={p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')}))
rows=json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else [];done={(r['method'],r['video'],r['source_sha256']) for r in rows}
v=vae();vae_sha=sha(MODEL);assert vae_sha=='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
with torch.no_grad():
    for idx,record in enumerate(records):
        if all((name,record['path'],digest) in done for name,m,c,ck,digest in configs):continue
        assert sha(record['path'])==input_hashes[record['path']],'historical input changed'
        x=tensor(record);reference=rgb(decode(v,x));reference_yuv=yuv(reference);canonical=None
        for name,m,c,ck,digest in configs:
            frozen=name.startswith('D42_temporal5000')
            if frozen and canonical is None:
                baseq=torch.cat([m.base.ga(f[None]).round() for f in x]);basez=torch.cat([m.base.gs(q[None]) for q in baseq]);canonical=(baseq,basez)
                atomic_torch(out/f'canonical_{idx:02d}.pt',dict(q=baseq.to(torch.int16),z=basez,video=record['path']))
            if (name,record['path'],digest) in done:continue
            leaf=out/f'{idx:02d}_{name}';leaf.mkdir(exist_ok=True);stream=leaf/'video.bin';fixed=m.fixed_sequence(x)
            stats=m.compress_fixed(fixed,list(x.shape[-2:]),stream,digest);q,z=m.decompress_video(stream,digest)
            assert torch.equal(q,torch.cat([f[0] for f in fixed])) and torch.equal(z,torch.cat([m.base.gs(f[0]) for f in fixed]))
            if frozen:assert torch.equal(q,canonical[0]) and torch.equal(z,canonical[1])
            if idx==0:
                target=leaf/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(stream),str(target)],check=True)
                independent=torch.load(target,weights_only=True);assert torch.equal(independent['q'],q) and torch.equal(independent['z'],z)
            image=rgb(decode(v,z));quality_row=quality(reference,image,reference_yuv)
            score=ssim(torch.from_numpy(reference.copy()).permute(0,3,1,2).float(),torch.from_numpy(image.copy()).permute(0,3,1,2).float(),data_range=255,size_average=True).item()
            main=sum(s['main'] for s in stats['slices']);hyper=sum(s['hyper'] for s in stats['slices']);assert main+hyper+stats['metadata_bytes']==stats['actual_bytes']
            row=dict(method=name,video=record['path'],source_sha256=digest,input_sha256=sha(record['path']),actual_bytes=stats['actual_bytes'],main_bytes=main,hyper_bytes=hyper,metadata_bytes=stats['metadata_bytes'],stream_sha256=[stats['sha256']],slice_bytes=stats['slices'],rate_diagnostics=temporal_rates(m,fixed,stats['slices']),latent_mse=(z-x).square().mean().item(),latent_mse_per_time_slice=(z-x).square().mean((1,2,3)).tolist(),**quality_row,SSIM_RGB=score,bpp=stats['actual_bytes']*8/(832*480*17),kbps=stats['actual_bytes']*8/1000/(17/16),real_decode=True,independent_process_decode=idx==0,fixed_symbol_control=frozen,canonical_symbols_exact=True if frozen else 'NOT APPLICABLE',reconstruction_exact=True if frozen else 'NOT APPLICABLE',training_step=c['step'],training_videos=len(c['train_paths']),scope='all18 historical development; independent generalization NOT VERIFIED',LPIPS='NOT RUN',DISTS='NOT RUN')
            row['training_lambda_z']=c.get('lambda_z');row['training_source_sha256']=c.get('source_sha256')
            rows.append(row);done.add((name,record['path'],digest));save(out/'rows.json',rows)
            print(json.dumps({k:row[k] for k in ['method','video','actual_bytes','psnr_y']}),flush=True)
assert len(rows)==18*len(configs) and sha(MODEL)==vae_sha
save(out/'status.json',dict(status='COMPLETE',videos=18,configurations=len(configs),rows=len(rows),vae_file_unchanged=True,sealed_test='NOT OPENED'))
