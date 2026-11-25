"""Isolated representative latency decomposition; run after quality workers finish."""
import json
import time
import importlib.util
import struct
import numpy as np
import runtime as r
from run import ROOT,E0,MODEL,save
from codec.lvc import encode,decode

def stats(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values,ddof=1)),min=min(values),max=max(values),samples=values)

def main():
    assert (ROOT/'results/ALL_MAIN_MEASUREMENTS_COMPLETE').exists()
    r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0');rows=[];h264_rows=[]
    spec=importlib.util.spec_from_file_location('e0_codec_baseline',E0/'codec/lvc.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    for source in r.samples:
        name=source['sample'];z=r.torch.load(source['latent'],weights_only=True,map_location='cpu').numpy()[None]
        paths=[ROOT/'results/E1_0'/name/f'Q{bits}.lvc' for bits in (4,5,6,7,8)]
        mixed=json.loads((ROOT/'results/E1_2/mixed_precision_results.json').read_text())
        candidates=[v for v in mixed if v['sample']==name and v['policy']['method'] in ('M2','M2_three','M3')]
        best=max(candidates,key=lambda v:v['psnr_y'])
        paths.append(ROOT/'results/E1_2'/name/('bits_'+''.join(map(str,best['channel_bits']))+'.lvc'))
        for stream in paths:
            blob=stream.read_bytes();n=struct.unpack('<I',blob[4:8])[0];h=json.loads(blob[8:8+n])
            if h.get('mixed'):
                offset=8+n;size=struct.unpack('<I',blob[offset+4:offset+8])[0];child=json.loads(blob[offset+8:offset+8+size])
                cfg=dict(channel_bits=h['channel_bits']);comp=child['compression'];meta=child['video']
            else:cfg=h['quantization'];comp=h['compression'];meta=h['video']
            profiles=[];target=stream.with_suffix('.profile.lvc')
            for i in range(6):
                profile={};encode(z,target,cfg,comp,meta,profile)
                assert target.read_bytes()==blob, 'profiling changed serialized bytes'
                if i:profiles.append(profile)
            _,legacy_timing=r.timing(lambda:legacy.encode(z,target,cfg,comp,meta))
            assert target.read_bytes()==blob
            gpu=r.torch.from_numpy(z).to('cuda');gpu_times=[];transfer_times=[]
            for i in range(6):
                r.torch.cuda.synchronize();start=time.perf_counter()
                host=gpu.cpu().numpy();copied=time.perf_counter()
                encode(host,target,cfg,comp,meta);r.torch.cuda.synchronize();elapsed=time.perf_counter()-start
                assert target.read_bytes()==blob
                if i:gpu_times.append(elapsed);transfer_times.append(copied-start)
            zh,dt=r.timing(lambda:decode(target))
            _,vt=r.timing(lambda:r.pixels(zh))
            timing={k:stats([v[k] for v in profiles]) for k in profiles[0]}
            rows.append(dict(sample=name,stream=str(stream.relative_to(ROOT)),timings=timing,decode_s=dt,vae_s=vt,gpu_latent_to_bitstream_s=stats(gpu_times),d2h_s=stats(transfer_times),e0_same_config_encode_s=legacy_timing,extra_encode_mean_s=timing['total_s']['mean']-legacy_timing['mean'],post_dit_two_ends_mean_s=float(np.mean(gpu_times))+dt['mean']+vt['mean'],scope='Saved original final latent placed on GPU; no DiT rerun; synchronized GPU-to-host plus CPU encoding; I/O OS cache, no fsync; native VAE GPU sync; preexisting GPU services remain; E0 comparison uses identical config/metadata and asserts identical bytes'))
            target.unlink()
        path=ROOT/'results/E1_0'/name/'baseline_bench.264';baseline_times=[];receiver_times=[]
        for i in range(6):
            r.torch.cuda.synchronize();start=time.perf_counter()
            with r.torch.no_grad():video=r.vae.decode([gpu[0]])[0]
            r.torch.cuda.synchronize()
            rgb=((video.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy();raw=r.yuv(rgb)
            r.cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-preset','medium','-qp','28','-g','17','-bf','2','-refs','3','-f','h264',str(path)],raw)
            end=time.perf_counter()
            decoded=r.cmd(['ffmpeg','-v','error','-i',str(path),'-pix_fmt','yuv420p','-f','rawvideo','pipe:1']);r.torch.cuda.synchronize();full=time.perf_counter()
            if i:baseline_times.append(end-start);receiver_times.append(full-start)
        psnr_y=r.quality(rgb,rgb,raw,decoded)['psnr_y']
        h264_rows.append(dict(sample=name,method='x264_medium_QP28',post_dit_encoding_s=stats(baseline_times),post_dit_two_ends_s=stats(receiver_times),total_bytes=path.stat().st_size,psnr_y=psnr_y,scope='Original final latent on GPU -> native VAE -> RGB transfer -> BT709 YUV -> x264; receiver raw YUV; fixed QP28 comparison is not quality-matched'))
    save(ROOT/'results/h264_post_dit_benchmark.json',h264_rows)
    save(ROOT/'results/latency_breakdown.json',rows)

if __name__=='__main__':main()
