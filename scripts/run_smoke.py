import sys,os,json,hashlib,time,subprocess,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from codec.lvc import encode,decode,pack,unpack
sys.path.insert(0,str(Path.home()/'Wan2.1'))
from wan.modules.vae import WanVAE
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
BASE=Path.home()/'experiments/t2v_pipeline_baseline.VN08gB'
samples=json.loads((BASE/'configs/samples.json').read_text())
torch.set_num_threads(4)
def sha(b): return hashlib.sha256(b).hexdigest()
def cmd(args,data=None):
    with (ROOT/'logs/commands.jsonl').open('a') as f: f.write(json.dumps(args)+'\n')
    return subprocess.run(args,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
def timing(fn):
    vals=[]
    for i in range(6):
        t=time.perf_counter(); v=fn(); elapsed=time.perf_counter()-t
        if i: vals.append(elapsed)
    return v,dict(mean=float(np.mean(vals)),median=float(np.median(vals)),std=float(np.std(vals,ddof=1)),min=min(vals),max=max(vals),samples=vals)
def pixels(z):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
    with torch.no_grad(): v=vae.decode([torch.from_numpy(z[0]).to('cuda')])[0]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-t
    peak=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
    rgb=((v.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
    return rgb,elapsed,peak,reserved
def yuv(rgb):
    return cmd(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'],rgb.tobytes())
def quality(a,b,ya,yb):
    mse=((a.astype('float64')-b)**2).mean(axis=(1,2,3)); y1=np.frombuffer(ya,dtype='uint8').reshape(17,-1)[:,:832*480].astype('float64'); y2=np.frombuffer(yb,dtype='uint8').reshape(17,-1)[:,:832*480]
    ym=((y1-y2)**2).mean(axis=1)
    ps=lambda m: float(10*np.log10(255**2/max(float(m),1e-30)))
    return dict(psnr_rgb=ps(mse.mean()),psnr_y=ps(ym.mean()),per_frame_psnr_y=[ps(m) for m in ym],worst_frame_psnr_y=ps(max(ym)),last4_psnr_y=float(np.mean([ps(m) for m in ym[-4:]])),lpips='NOT RUN: package/weights unavailable')
for bits in (2,3,4,5,6,8):
    q=np.arange(-(1<<(bits-1)),1<<(bits-1),dtype='int16'); q=np.tile(q,3)[:-1]
    assert np.array_equal(q,unpack(pack(q,bits),bits,len(q)))
(ROOT/'results/packing_gate.json').write_text(json.dumps({'status':'PASSED','bits':[2,3,4,5,6,8],'odd_count':True}))
t=time.perf_counter(); vae=WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0'); load_s=time.perf_counter()-t
allrows=[]
for s in samples:
    folder=OUT/s['sample']; folder.mkdir(exist_ok=True)
    assert sha(Path(s['latent']).read_bytes())==s['latent_sha256']
    z=torch.load(s['latent'],weights_only=True,map_location='cpu'); assert z.dtype==torch.float32
    z=z.numpy()[None]; meta=dict(s,vae='Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',normalization='WanVAE.decode applies self.scale internally; solver final latent unchanged')
    stats=dict(shape=list(z.shape),dtype=str(z.dtype),bytes=z.nbytes,sha256=sha(z.tobytes()),min=float(z.min()),max=float(z.max()),mean=float(z.mean()),std=float(z.std()),nan=int(np.isnan(z).sum()),inf=int(np.isinf(z).sum()),histogram=[v.tolist() for v in np.histogram(z,bins=100)],channel_mean=z.mean(axis=(0,2,3,4)).tolist(),channel_std=z.std(axis=(0,2,3,4)).tolist(),time_mean=z.mean(axis=(0,1,3,4)).tolist(),time_std=z.std(axis=(0,1,3,4)).tolist(),temporal_corr=float(np.corrcoef(z[:,:,:-1].ravel(),z[:,:,1:].ravel())[0,1]),horizontal_corr=float(np.corrcoef(z[:,:,:,:,:-1].ravel(),z[:,:,:,:,1:].ravel())[0,1]),vertical_corr=float(np.corrcoef(z[:,:,:,:-1].ravel(),z[:,:,:,1:].ravel())[0,1]),channel_corr=np.corrcoef(z.transpose(1,0,2,3,4).reshape(z.shape[1],-1)).tolist())
    (folder/'statistics.json').write_text(json.dumps(stats,indent=2))
    pixels(z) # warmup
    ref,rt=timing(lambda:pixels(z)); rgb,_,peak,reserved=ref; raw=yuv(rgb)
    assert sha(raw)==sha(Path(s['final_yuv']).read_bytes()), 'REF gate historical mismatch'
    (folder/'reference.rgb').write_bytes(rgb.tobytes()); (folder/'reference.yuv').write_bytes(raw)
    (folder/'reference.json').write_text(json.dumps(dict(historical_match=True,vae_timing=rt,peak_allocated=peak,peak_reserved=reserved,model_load_s=load_s,metadata=meta)))
    for preset in ('medium','fast'):
        for qp in (18,22,26,28,32,37,42):
            path=folder/f'x264_{preset}_{qp}.264'
            def enc(): return cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i',str(folder/'reference.yuv'),'-c:v','libx264','-preset',preset,'-qp',str(qp),'-g','17','-bf','2','-refs','3','-f','h264',str(path)])
            _,et=timing(enc)
            rr,dt=timing(lambda:cmd(['ffmpeg','-v','error','-i',str(path),'-pix_fmt','yuv420p','-f','rawvideo','pipe:1']))
            # RGB metrics require matching inverse BT709 conversion; mark missing rather than mixing conventions.
            yy=np.frombuffer(raw,dtype='uint8').reshape(17,-1)[:,:832*480].astype('float64'); yy2=np.frombuffer(rr,dtype='uint8').reshape(17,-1)[:,:832*480]; mse=np.mean((yy-yy2)**2)
            row=dict(sample=s['sample'],method=f'x264_{preset}_{qp}',total_bytes=path.stat().st_size,bitrate_kbps=path.stat().st_size*8/(17/16)/1000,psnr_y=float(10*np.log10(255**2/mse)),encode_s=et,decode_s=dt,psnr_rgb='NOT RUN',ssim='NOT RUN',lpips='NOT RUN')
            allrows.append(row); (OUT/'rows.json').write_text(json.dumps(allrows,indent=2)); print(s['sample'],row['method'],row['total_bytes'],flush=True)
    configurations=[('A0',{},'none',False),('A1',{},'zlib',False),('A2',{},'zstd',False),('A3',{},'zstd',True)]
    for bits,letter in [(8,'B'),(4,'C'),(2,'D')]:
        for k in range(3): configurations.append((letter+str(k),dict(bits=bits,granularity='channel' if k==2 else 'tensor'), 'none' if k==0 else 'zstd',False))
    for bits in (8,4,2):
        for gran in ('tensor','channel'):
            for pct in (99,99.5,99.9): configurations.append((f'Q{bits}_{gran}_{pct}',dict(bits=bits,granularity=gran,percentile=pct),'zstd',False))
    configurations.append(('E1',dict(bits=4,granularity='channel',temporal='integer_delta'),'zstd',False))
    for refresh in (2,4,8,0): configurations.append((f'E2_refresh{refresh}',dict(bits=4,granularity='channel',temporal='closed_loop',refresh=refresh),'zstd',False))
    for name,q,alg,shuffle in configurations:
        path=folder/(name+'.lvc'); sizes,et=timing(lambda:encode(z,path,q,dict(algorithm=alg,shuffle=shuffle),meta)); zh,dt=timing(lambda:decode(path))
        if not q: assert z.dtype==zh.dtype and z.tobytes()==zh.tobytes(); rec=rgb; rraw=raw; vt=0
        else: rec,vt,_,_=pixels(zh); rraw=yuv(rec)
        if name=='E1': assert np.array_equal(zh,decode(folder/'C2.lvc'))
        diff=z.astype('float64')-zh
        row=dict(sample=s['sample'],method=name,quantization=q,**sizes,bitrate_kbps=sizes['total_bytes']*8/(17/16)/1000,bpp=sizes['total_bytes']*8/(17*480*832),compression_ratio=z.nbytes/sizes['total_bytes'],encode_s=et,decode_s=dt,receiver_vae_s=vt,latent_mse=float(np.mean(diff**2)),latent_mae=float(np.mean(abs(diff))),latent_cosine=float(np.dot(z.ravel(),zh.ravel())/(np.linalg.norm(z)*np.linalg.norm(zh))),**quality(rgb,rec,raw,rraw))
        allrows.append(row); (OUT/'rows.json').write_text(json.dumps(allrows,indent=2)); print(s['sample'],name,row['total_bytes'],row['psnr_y'],flush=True)
        if name in ('B2','C2','D2','E1'):
            cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-crf','18',str(folder/(name+'.mp4'))],rec.tobytes())
(OUT/'COMPLETE').touch()
