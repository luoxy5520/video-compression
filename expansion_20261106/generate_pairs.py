"""Standalone instrumentation of upstream Wan; leaves upstream source unchanged."""
import sys, json, subprocess, time, hashlib, shutil
from pathlib import Path
import torch
ROOT=Path('/home/luoxiangyu/Wan2.1')
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/baseline17')]
from run_remote_baseline17 import sdpa
import wan.modules.model as wm
from wan.text2video import WanT2V, FlowUniPCMultistepScheduler
from wan.configs import WAN_CONFIGS
wm.flash_attention=sdpa
torch.set_num_threads(4)
cfg=json.loads(Path(sys.argv[1]).read_text())
out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=True)
(out/'config.json').write_text(json.dumps(cfg,indent=2))
pipe=WanT2V(WAN_CONFIGS['t2v-1.3B'],cfg['checkpoint'],t5_cpu=True)
original=FlowUniPCMultistepScheduler.step
state={}
def hook(module,args,output):
    if state['i']+1 in cfg['observe'] and state['calls']%2==0:
        state['hidden']=output.detach().cpu().clone()
    state['calls']+=1
handle=pipe.model.blocks[-1].register_forward_hook(hook)
def step(self,model_output,timestep,sample,*args,**kwargs):
    i=state['i']+1
    if i in cfg['observe']:
        sigma=self.sigmas[state['i']].to(sample.device)
        clean=(sample-sigma*model_output).squeeze(0).detach().cpu()
        p=state['dir']/f'step_{i:03d}'; p.mkdir()
        torch.save(clean,p/'clean_latent.pt')
        torch.save(state['hidden'],p/'hidden_cond_last_block.pt')
        state['records'].append(dict(sample_id=state['id'],step=i,prompt=state['prompt'],seed=state['seed'],latent=str(p/'clean_latent.pt'),hidden=str(p/'hidden_cond_last_block.pt'),raw_yuv=str(p/'raw.yuv'),timestep=float(timestep),sigma=float(sigma),width=832,height=480,frames=17,fps=16,pixel_format='yuv420p',hidden_branch='conditional',hidden_block=29))
    result=original(self,model_output,timestep,sample,*args,**kwargs)
    state['i']+=1
    if state['i']==cfg['steps']: torch.save(result[0].squeeze(0).cpu(),state['dir']/'final_latent.pt')
    return result
FlowUniPCMultistepScheduler.step=step
for idx,item in enumerate(cfg['samples']):
    sid=item.get('sample_id',f'sample_{idx:04d}'); folder=out/sid
    if (folder/'complete.json').exists(): continue
    folder.mkdir(exist_ok=True)
    if list(folder.glob('step_*')):
        folder.rename(out/(sid+'_incomplete_'+str(int(time.time())))); folder.mkdir()
    state.update(i=0,calls=0,dir=folder,id=sid,prompt=item['prompt'],seed=item['seed'],records=[])
    start=time.time()
    final_video=pipe.generate(item['prompt'],size=(832,480),frame_num=17,shift=8,sample_solver='unipc',sampling_steps=cfg['steps'],guide_scale=6,seed=item['seed'],offload_model=True)
    for r in state['records']:
        z=torch.load(r['latent'],weights_only=True).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): v=pipe.vae.decode([z])[0]
        rgb=((v.float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy().tobytes()
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709','-pix_fmt','yuv420p','-f','rawvideo',r['raw_yuv']],input=rgb,check=True)
        assert Path(r['raw_yuv']).stat().st_size==832*480*17*3//2
        r.update(checkpoint=cfg['checkpoint'],sampler='unipc',total_steps=20,color_matrix='bt709',color_range='limited',final_latent=str(folder/'final_latent.pt'))
        r.update(category=item['category'],split=item['split'])
        del z,v
    final_rgb=((final_video.float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy().tobytes()
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709','-pix_fmt','yuv420p','-f','rawvideo','-y',str(folder/'final_raw.yuv')],input=final_rgb,check=True)
    existing=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()] if (out/'manifest.jsonl').exists() else []
    existing=[r for r in existing if r['sample_id']!=sid]+state['records']
    temp=out/'manifest.tmp'; temp.write_text(''.join(json.dumps(r)+'\n' for r in existing)); temp.replace(out/'manifest.jsonl')
    del final_video, final_rgb
    (folder/'complete.json').write_text(json.dumps({'seconds':time.time()-start,'records':len(state['records'])}))
    print(sid,'COMPLETE',flush=True)
