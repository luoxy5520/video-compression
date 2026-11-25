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
