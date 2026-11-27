from pixels import *
from temporal_entropy_model import load_temporal
ck=ROOT/'checkpoints/temporal_joint/P2.pt';m,c=load_temporal(ck);out=ROOT/'results/E2_2/P2';records=json.loads((E2/'dataset_manifest.json').read_text())['records'];groups={r['group'] for r in records if r['path'] in c['train_paths']};dev=[r for r in records if r['role']=='historical_development' and r['group'] not in groups][:2];v=vae();rows=[]
for idx,r in enumerate(dev):
    x=tensor(r);path=out/f'dev{idx}.bin';stats=m.compress_video(x,path,sha(ck));q,z=m.decompress_video(path,sha(ck));expected=torch.cat([m.base.gs(f[0]) for f in m.fixed_sequence(x)]);assert torch.equal(z,expected)
    external=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(path),str(external)],check=True);d=torch.load(external,weights_only=True);assert torch.equal(d['q'],q) and torch.equal(d['z'],z)
    with torch.no_grad():ref=rgb(decode(v,x));img=rgb(decode(v,z));quality_row=quality(ref,img)
    rows.append(dict(method='P2_joint_latent_RD',video=r['path'],actual_bytes=stats['actual_bytes'],streams=stats,latent_mse=(z-x).square().mean().item(),**quality_row,independent_decode=True,fixed_symbols='NOT APPLICABLE: joint lossy training',checkpoint_sha256=sha(ck),scope='historical development only'));save(out/'development.json',rows);print(json.dumps(rows[-1]),flush=True)
