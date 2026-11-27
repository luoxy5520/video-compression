from common import *
import numpy as np,subprocess
sys.path.insert(0,str(Path.home()/'Wan2.1'))
from wan.modules.vae import WanVAE
MODEL=Path('/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
def vae():return WanVAE(vae_pth=str(MODEL),device='cuda:0')
def rgb(v):return ((v.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
def yuv(im):
    return subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'],input=im.tobytes(),capture_output=True,check=True).stdout
def quality(a,b,ya=None):
    ya=yuv(a) if ya is None else ya;yb=yuv(b);ay=np.frombuffer(ya,dtype=np.uint8).reshape(17,-1)[:,:832*480].astype(float);by=np.frombuffer(yb,dtype=np.uint8).reshape(17,-1)[:,:832*480].astype(float);ym=((ay-by)**2).mean(1);rm=((a.astype(float)-b)**2).mean((1,2,3));ps=lambda mse:10*np.log10(255**2/max(float(mse),1e-30))
    return dict(psnr_y=ps(ym.mean()),psnr_rgb=ps(rm.mean()),per_frame_psnr_y=[ps(m) for m in ym],last4_psnr_y=float(np.mean([ps(m) for m in ym[-4:]])))
def decode(v,x):return v.decode([x.permute(1,0,2,3).to('cuda')])[0]
