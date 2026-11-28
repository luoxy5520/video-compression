"""CPU-only actual-stream audit of completed 20k checkpoints on declared development.

No pixel quality is inferred from latent error. Full frozen-VAE evaluation is separate.
"""
from common import *
from diagnostic_codec import load

out=ROOT/'results/extended_checkpoint_audit';out.mkdir(exist_ok=True)
records=[r for r in json.loads((E2/'dataset_manifest.json').read_text())['records']
         if r['role']=='historical_development' and r['exists']]
assert len(records)==18
rows=json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else []
done={(r['checkpoint_sha256'],r['video']) for r in rows}
for folder in ('D16_f1_seed20261127','D42_f1_seed20261127'):
    ck=ROOT/'checkpoints'/folder/'step20000.pt';model,state=load(ck)
    assert state['step']==20000
    digest=sha(ck)
    for idx,record in enumerate(records):
        if (digest,record['path']) in done:continue
        x=tensor(record);decoded=[];stats=[]
        leaf=out/f'{folder}_{idx:02d}';leaf.mkdir(exist_ok=True)
        with torch.no_grad():
            for t,frame in enumerate(x):
                stream=leaf/f't{t}.lvc2'
                stats.append(write_stream(stream,model,frame[None],digest))
                rec=read_stream(stream,model,digest);expected,_=model(frame[None])
                assert torch.equal(rec,expected) and torch.isfinite(rec).all()
                decoded.append(rec)
            z=torch.cat(decoded)
        if idx==0:
            target=leaf/'independent.pt'
            subprocess=__import__('subprocess')
            subprocess.run([sys.executable,str(ROOT/'scripts/diagnostic_codec.py'),
                            str(ck),str(leaf/'t0.lvc2'),str(target)],check=True)
            assert torch.equal(torch.load(target,weights_only=True),decoded[0])
        row=dict(checkpoint=folder,step=20000,checkpoint_sha256=digest,
                 video=record['path'],input_sha256=sha(record['path']),
                 actual_bytes=sum(s['bytes'] for s in stats),
                 latent_mse=(z-x).square().mean().item(),
                 latent_mse_per_time_slice=(z-x).square().mean((1,2,3)).tolist(),
                 stream_sha256=[s['sha256'] for s in stats],real_decode=True,
                 first_video_first_slice_independent_decode=(idx==0),
                 pixel_quality='NOT MEASURED: CPU-only audit',
                 scope='18 declared historical development videos; sealed test not read')
        rows.append(row);done.add((digest,record['path']));save(out/'rows.json',rows)
        print(json.dumps(row),flush=True)
assert len(rows)==36
csvsave(out/'summary.csv',[dict(checkpoint=folder,videos=18,
        mean_actual_bytes=sum(r['actual_bytes'] for r in rows if r['checkpoint']==folder)/18,
        mean_latent_mse=sum(r['latent_mse'] for r in rows if r['checkpoint']==folder)/18)
        for folder in ('D16_f1_seed20261127','D42_f1_seed20261127')])
save(out/'status.json',dict(status='PASS',rows=len(rows),historical_videos=18,
     checkpoint_count=2,pixel_quality='NOT MEASURED',sealed_test='NOT OPENED'))
