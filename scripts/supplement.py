# Run only after smoke passes. Reuse helpers without executing main experiment.
from pathlib import Path
source=Path(__file__).with_name('run_smoke.py').read_text(); exec(source[:source.index('for bits in (2,3,4,5,6,8):')])
assert (OUT/'COMPLETE').exists()
vae=WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
rows=[]
for s in samples:
    folder=OUT/s['sample']; z=torch.load(s['latent'],weights_only=True,map_location='cpu').numpy()[None]
    rgb=np.fromfile(folder/'reference.rgb',dtype='uint8').reshape(17,480,832,3); raw=(folder/'reference.yuv').read_bytes()
    sensitivity=[]
    for c in range(z.shape[1]):
        tmp=folder/'channel_tmp.lvc'; encode(z[:,c:c+1],tmp,dict(bits=4,granularity='channel'),dict(algorithm='zstd'),{})
        zh=z.copy(); zh[:,c:c+1]=decode(tmp); rec,_,_,_=pixels(zh); metrics=quality(rgb,rec,raw,yuv(rec)); sensitivity.append(metrics['psnr_y'])
    sensitive=np.argsort(sensitivity)[:4]; bits=[8 if c in sensitive else 4 for c in range(z.shape[1])]
    path=folder/'mixed.lvc'; sizes,et=timing(lambda:encode(z,path,dict(channel_bits=bits),dict(algorithm='zstd'),s)); zh,dt=timing(lambda:decode(path)); rec,vt,_,_=pixels(zh)
    rows.append(dict(sample=s['sample'],method='F_mixed',channel_single_int4_psnr_y=sensitivity,channel_bits=bits,**sizes,encode_s=et,decode_s=dt,**quality(rgb,rec,raw,yuv(rec))))
    cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-crf','18',str(folder/'mixed.mp4')],rec.tobytes())
    # Transfer experiment: actual synchronized CPU and GPU quantization, no VAE in encoding path.
    gpu=torch.from_numpy(z).cuda(); torch.cuda.synchronize()
    def cpu_impl():
        torch.cuda.synchronize(); t=time.perf_counter(); host=gpu.cpu(); torch.cuda.synchronize(); transfer=time.perf_counter()-t
        t=time.perf_counter(); sc=host.abs().amax()/127; q=(host/sc).round().clamp(-127,127).to(torch.int8); return dict(transfer_s=transfer,quantization_s=time.perf_counter()-t,transfer_bytes=host.numel()*host.element_size())
    def gpu_impl():
        torch.cuda.synchronize(); t=time.perf_counter(); sc=gpu.abs().amax()/127; q=(gpu/sc).round().clamp(-127,127).to(torch.int8); torch.cuda.synchronize(); quant=time.perf_counter()-t
        t=time.perf_counter(); host=q.cpu(); sc.cpu(); torch.cuda.synchronize(); return dict(quantization_s=quant,transfer_s=time.perf_counter()-t,transfer_bytes=host.numel()+4)
    a,at=timing(cpu_impl); b,bt=timing(gpu_impl)
    (folder/'transfer.json').write_text(json.dumps(dict(A=a,A_timing=at,B=b,B_timing=bt,cpu_rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actual_memory_traffic='NOT RUN: no hardware profiling'),indent=2))
    # Existing JM encoder configurations and binary reused unchanged.
    sys.path.insert(0,str(BASE/'scripts')); import profile_encoder as jm
    for qp in (18,22,26,28,32,37,42):
        dest=folder/f'jm_{qp}'; dest.mkdir(exist_ok=True); jm.PARAMS.update(QPISlice=qp,QPPSlice=qp)
        def runjm():
            (dest/'video.mp4').unlink(missing_ok=True)
            tt={}; jm.encode(dest,folder/'reference.yuv',tt); return tt
        result,et=timing(runjm); rr=(dest/'recon.yuv').read_bytes(); yy=np.frombuffer(raw,dtype='uint8').reshape(17,-1)[:,:832*480].astype('float64'); yy2=np.frombuffer(rr,dtype='uint8').reshape(17,-1)[:,:832*480]
        rows.append(dict(sample=s['sample'],method=f'JM_{qp}',total_bytes=(dest/'stream.264').stat().st_size,psnr_y=float(10*np.log10(255**2/np.mean((yy-yy2)**2))),encode_with_mux_s=et,last_stage_times=result))
        (OUT/'supplement.json').write_text(json.dumps(rows,indent=2))
(OUT/'SUPPLEMENT_COMPLETE').touch()
