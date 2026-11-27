"""Frozen configurations on all18 historical development videos; never sealed test.
Broader development coverage is not independent validation. Completed rows skip on resume.
"""
from pixels import *
from diagnostic_codec import load
from temporal_entropy_model import load_temporal
from pytorch_msssim import ssim
import argparse,importlib.util
p=argparse.ArgumentParser();p.add_argument('--include-p3',action='store_true');a=p.parse_args()
out=ROOT/'results/development_pool';out.mkdir(exist_ok=True);deployed=out/'deploy';deployed.mkdir(exist_ok=True)
uniform_path=E2.parent/'experiment_1/codec/lvc.py'
spec=importlib.util.spec_from_file_location('pool_uniform',uniform_path);uniform=importlib.util.module_from_spec(spec);spec.loader.exec_module(uniform)
configs=[(f'Uniform_INT{bits}',bits,uniform_path,'uniform') for bits in range(2,9)]
for lam in (16,128,1024,8192):
    ck=E2/'checkpoints'/f'I3_lambda{lam}.pt';m=load_model(ck);configs.append((f'historical_I3_lambda{lam}',m,ck,'historical'))
for factor,step in ((.1,5000),(1,10000),(10,5000)):
    ck=ROOT/'checkpoints'/f'D16_f{factor:g}_seed20261127'/f'step{step}.pt';m,c=load(ck);configs.append((f'trained_I3_f{factor:g}',m,ck,'intra'))
for name,folder,step in [('trained_I3_mid_1000','D16_f1_seed20261127',1000),('trained_I3_mid_5000','D16_f1_seed20261127',5000),('trained_I3_mid_seed2_5000','D16_f1_seed20261128',5000),('trained_I3_D42_mid_5000','D42_f1_seed20261127',5000)]:
    ck=ROOT/'checkpoints'/folder/f'step{step}.pt';m,c=load(ck);configs.append((name,m,ck,'intra'))
for mode in ('V1','V2'):
    ck=ROOT/'checkpoints'/f'{mode}_D16/step1000.pt';m,c=load(ck);configs.append((mode,m,ck,'intra'))
for name in ('V2_low_D16','V2_high_D16'):
    ck=ROOT/'checkpoints'/name/'step1000.pt';m,c=load(ck);configs.append((name,m,ck,'intra'))
for folder in ('D16_f1_seed20261127','D42_f1_seed20261127'):
    ck=ROOT/'checkpoints'/folder/'step20000.pt'
    if ck.exists():
        m,c=load(ck);configs.append((folder+'_20000',m,ck,'intra'))
ck=ROOT/'checkpoints/V2_D16/step3000.pt'
if ck.exists():
    m,c=load(ck);configs.append(('V2_mid_3000',m,ck,'intra'))
for mode in ('TP0','TP1','TP2','TP2Z'):
    ck=ROOT/'checkpoints/temporal'/f'{mode}.pt';m,c=load_temporal(ck);configs.append((mode,m,ck,'temporal'))
ck=ROOT/'checkpoints/temporal_joint/P2.pt';m,c=load_temporal(ck);configs.append(('P2',m,ck,'temporal_joint'))
if a.include_p3:
    ck=ROOT/'checkpoints/temporal_pixel/P3.pt';m,c=load_temporal(ck);configs.append(('P3',m,ck,'temporal_joint'))
records=[r for r in json.loads((E2/'dataset_manifest.json').read_text())['records'] if r['role']=='historical_development' and r['exists']];assert len(records)==18
input_digests={r['path']:r['file_sha256'] for r in json.loads((E2.parent/'experiment_1/results/input_manifest.json').read_text())}
rows=json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else [];done={(r['method'],r['video'],r['source_sha256']) for r in rows};v=vae();base=next(m for name,m,ck,kind in configs if name=='trained_I3_f1')
for ridx,r in enumerate(records):
    pending=[c for c in configs if (c[0],r['path'],sha(c[2])) not in done]
    if not pending:continue
    assert sha(r['path'])==input_digests[r['path']],'historical input changed'
    x=tensor(r);assert list(x.shape)==[5,16,60,104],'unexpected frozen-pool shape'
    with torch.no_grad():
        ref=rgb(decode(v,x));ya=yuv(ref);baseq=torch.cat([base.ga(f[None]).round() for f in x]);basez=torch.cat([base.gs(q[None]) for q in baseq]);baseimg=rgb(decode(v,basez));basequality=quality(ref,baseimg,ya)
    basessim=ssim(torch.from_numpy(ref.copy()).permute(0,3,1,2).float(),torch.from_numpy(baseimg.copy()).permute(0,3,1,2).float(),data_range=255,size_average=True).item()
    for name,m,ck,kind in pending:
        modelpath=deployed/f'{name}.pt'
        if kind!='uniform' and not modelpath.exists():
            if kind=='historical':state=dict(kind='I3',channels=m.channels,state=m.state_dict())
            elif kind=='intra':state=dict(config=m.config,state=m.state_dict())
            else:state=dict(config=m.config,mode=m.mode,state=m.state_dict())
            state['source_sha256']=sha(ck);atomic_torch(modelpath,state)
        elif kind!='uniform':assert torch.load(modelpath,weights_only=False,map_location='cpu')['source_sha256']==sha(ck),'source changed: choose new run ID'
        leaf=out/f'{ridx:02d}_{Path(r["path"]).parent.name}_{name}';leaf.mkdir(exist_ok=True);stats=[]
        with torch.no_grad():
            if kind=='uniform':
                stream=leaf/'video.lvc';st=uniform.encode(x.permute(1,0,2,3)[None].numpy(),stream,dict(bits=m,granularity='channel',percentile=100),dict(algorithm='zstd'),dict(width=832,height=480,frames=17,fps=16));z=torch.from_numpy(uniform.decode(stream))[0].permute(1,0,2,3).contiguous();size=st['total_bytes'];main=st['payload_bytes'];hyper=0;metadata=st['header_bytes'];hashes=[sha(stream)]
                if ridx==0:
                    target=leaf/'independent.npy';program="import importlib.util,numpy as np,sys;s=importlib.util.spec_from_file_location('decoder',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);np.save(sys.argv[3],m.decode(sys.argv[2]))";subprocess.run([sys.executable,'-c',program,str(uniform_path),str(stream),str(target)],check=True);assert torch.equal(torch.from_numpy(np.load(target))[0].permute(1,0,2,3).contiguous(),z)
            elif kind.startswith('temporal'):
                stream=leaf/'video.bin';st=m.compress_video(x,stream,sha(modelpath));q,z=m.decompress_video(stream,sha(modelpath));expected=torch.cat([m.base.gs(f[0]) for f in m.fixed_sequence(x)]);assert torch.equal(z,expected);size=st['actual_bytes'];main=sum(s['main'] for s in st['slices']);hyper=sum(s['hyper'] for s in st['slices']);metadata=st['metadata_bytes'];hashes=[st['sha256']]
                if name in ('TP0','TP1','TP2','TP2Z'):assert torch.equal(q,baseq) and torch.equal(z,basez)
                if ridx==0:
                    target=leaf/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(modelpath),str(stream),str(target)],check=True);assert torch.equal(torch.load(target,weights_only=True)['z'],z)
            else:
                recs=[]
                for t,frame in enumerate(x):
                    stream=leaf/f't{t}.lvc2';stats.append(write_stream(stream,m,frame[None],sha(modelpath)));decoded=read_stream(stream,m,sha(modelpath));recs.append(decoded)
                    expected,_=m(frame[None]);assert torch.equal(decoded,expected)
                    if ridx==0 and t==0:
                        target=leaf/'independent.pt';entry=E2/'codec.py' if kind=='historical' else ROOT/'scripts/diagnostic_codec.py';subprocess.run([sys.executable,str(entry),str(modelpath),str(stream),str(target)],check=True);assert torch.equal(torch.load(target,weights_only=True),decoded)
                z=torch.cat(recs);size=sum(s['bytes'] for s in stats);main=sum(s['main_bytes'] for s in stats);hyper=sum(s['hyper_bytes'] for s in stats);metadata=sum(s['metadata_bytes'] for s in stats);hashes=[s['sha256'] for s in stats]
            if name in ('TP0','TP1','TP2','TP2Z','trained_I3_f1'):
                assert torch.equal(z,basez);quality_row=basequality;score=basessim
            else:
                image=rgb(decode(v,z));quality_row=quality(ref,image,ya);score=ssim(torch.from_numpy(ref.copy()).permute(0,3,1,2).float(),torch.from_numpy(image.copy()).permute(0,3,1,2).float(),data_range=255,size_average=True).item()
        row=dict(method=name,video=r['path'],scope='all18 historical development videos, NOT independent validation',actual_bytes=size,main_bytes=main,hyper_bytes=hyper,metadata_bytes=metadata,bpp=size*8/(832*480*17),kbps=size*8/1000/(17/16),latent_mse=(z-x).square().mean().item(),latent_mse_per_time_slice=(z-x).square().mean((1,2,3)).tolist(),**quality_row,SSIM_RGB=score,ssim_scope='pytorch_msssim RGB8-bit,data_range255,mean17frames',stream_sha256=hashes,source_sha256=sha(ck),deploy_sha256=sha(modelpath) if kind!='uniform' else 'NOT APPLICABLE',real_decode=True,independent_process_decode=(ridx==0),independent_check_scope='first video per checkpoint independently decoded; every row actual in-process stream decoded',LPIPS='NOT RUN: no verified weights',DISTS='NOT RUN: no verified weights')
        rows.append(row);done.add((name,r['path'],sha(ck)));save(out/'rows.json',rows);print(json.dumps({k:row[k] for k in ['method','video','actual_bytes','psnr_y','SSIM_RGB']}),flush=True)
save(out/'status.json',dict(status='COMPLETE',historical_videos=len(records),configurations=len(configs),rows=len(rows),independent_validation=False,sealed_test='NOT OPENED'))
