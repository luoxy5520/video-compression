from pathlib import Path
source=Path(__file__).with_name('run_smoke.py').read_text(); exec(source[:source.index('for bits in (2,3,4,5,6,8):')])
vae=WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
rows=[]
for s in samples:
    folder=OUT/s['sample']; z=torch.load(s['latent'],weights_only=True,map_location='cpu').numpy()[None]; gpu=torch.from_numpy(z).cuda(); torch.cuda.synchronize()
    for method in ('x264_medium_28','INT8','INT4'):
        samples_t=[]
        for repeat in range(6):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            if method.startswith('x264'):
                with torch.no_grad(): video=vae.decode([gpu[0]])[0]
                torch.cuda.synchronize(); t1=time.perf_counter()
                rgb=((video.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy(); raw=yuv(rgb); t2=time.perf_counter()
                path=folder/'bench.264'; cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-preset','medium','-qp','28','-g','17','-bf','2','-refs','3','-f','h264',str(path)],raw); end=time.perf_counter()
                cmd(['ffmpeg','-v','error','-i',str(path),'-pix_fmt','yuv420p','-f','rawvideo','pipe:1']); torch.cuda.synchronize(); full=time.perf_counter()
                stages=dict(vae_s=t1-start,pixel_prepare_transfer_conversion_s=t2-t1,encode_serialization_s=end-t2)
                del video,rgb,raw
            else:
                host=gpu.cpu().numpy(); torch.cuda.synchronize(); transfer=time.perf_counter(); path=folder/'bench.lvc'; encode(host,path,dict(bits=8 if method=='INT8' else 4,granularity='channel'),dict(algorithm='zstd'),s); end=time.perf_counter()
                zh=decode(path); pixels(zh); torch.cuda.synchronize(); full=time.perf_counter(); stages=dict(latent_d2h_s=transfer-start,codec_s=end-transfer)
            row=dict(sample=s['sample'],method=method,repeat=repeat,warmup=repeat==0,post_dit_encoding_s=end-start,post_dit_to_receiver_pixels_s=full-start,stages=stages,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            if repeat: samples_t.append(row)
        rows+=samples_t
    (OUT/'benchmark.json').write_text(json.dumps(rows,indent=2)); print(s['sample'],'benchmark done',flush=True)
(OUT/'BENCHMARK_COMPLETE').touch()
