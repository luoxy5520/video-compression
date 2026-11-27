from pixels import *
import argparse,importlib.util
p=argparse.ArgumentParser();p.add_argument('--variant',default='R1');a=p.parse_args();out=ROOT/'results/E2_1b'/a.variant
v=vae();x=tensor(data()[0]);rows=[]
with torch.no_grad():
    ref=rgb(decode(v,x));ya=yuv(ref)
    spec=importlib.util.spec_from_file_location('i0',E2.parent/'experiment_1/codec/lvc.py');i0=importlib.util.module_from_spec(spec);spec.loader.exec_module(i0)
    for bits in (6,7,8):
        path=ROOT/'results/E2_1b'/f'Uniform_INT{bits}.lvc';i0.encode(x.permute(1,0,2,3)[None].numpy(),path,dict(bits=bits,granularity='channel',percentile=100),dict(algorithm='zstd'),dict(width=832,height=480,frames=17,fps=16));rec=torch.from_numpy(i0.decode(path))[0].permute(1,0,2,3);q=quality(ref,rgb(decode(v,rec)),ya);rows.append(dict(variant=f'INT{bits}',step=0,mse=(rec-x).square().mean().item(),mae=(rec-x).abs().mean().item(),relative_l2=((rec-x).norm()/x.norm()).item(),**q))
    for path in sorted(out.glob('step*.json'),key=lambda p:int(p.stem[4:])):
        stats=json.loads(path.read_text());rec=torch.load(out/f"reconstruction_step{stats['step']}.pt",weights_only=True);q=quality(ref,rgb(decode(v,rec)),ya);stats.update(q);save(path,stats);rows.append({k:stats[k] for k in ['variant','step','mse','mae','relative_l2',*q]})
    csvsave(out/'reconstruction_only_results.csv',rows);save(out/'pixel_results.json',rows)
    final=rgb(decode(v,rec));comparison=np.concatenate([ref,final],axis=2)
    subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1664x480','-r','16','-i','pipe:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'reconstructed_video_comparison.mp4')],input=comparison.tobytes(),check=True)
print(json.dumps(rows),flush=True)
