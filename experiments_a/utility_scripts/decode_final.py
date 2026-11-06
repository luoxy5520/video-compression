"""Decode solver final latents for A0, without loading DiT or T5."""
import argparse,json,sys,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--wan-root',default='/home/luoxiangyu/Wan2.1');p.add_argument('--checkpoint',default='/mnt/data_hdd/models/Wan2.1-T2V-1.3B');a=p.parse_args()
sys.path.insert(0,a.wan_root)
from wan.modules.vae import WanVAE
root=Path(a.data_root); vae=WanVAE(vae_pth=str(Path(a.checkpoint)/'Wan2.1_VAE.pth'),device='cuda')
source=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]; rows=[]
for r in source:
    if r['step']!=20:continue
    folder=root/r['sample_id']; target=folder/'final_raw.yuv'
    if not target.exists():
        z=torch.load(folder/'final_latent.pt',map_location='cuda',weights_only=True)
        with torch.no_grad():v=vae.decode([z])[0]
        rgb=((v.float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy().tobytes()
        temp=target.with_suffix('.yuv.partial')
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709','-pix_fmt','yuv420p','-f','rawvideo',str(temp)],input=rgb,check=True)
        assert temp.stat().st_size==10183680;temp.rename(target)
    assert target.stat().st_size==10183680
    rows.append(dict(sample_id=r['sample_id'],step=20,latent=str(folder/'final_latent.pt'),raw_yuv=str(target),prompt=r['prompt'],seed=r['seed'],latent_kind='solver_final',label_path=None,width=832,height=480,frames=17,fps=16,pixel_format='yuv420p',color_matrix='bt709',color_range='limited'))
    print(r['sample_id'],'final raw ready',flush=True)
(root/'a0_manifest.json').write_text(json.dumps(rows,indent=2))
