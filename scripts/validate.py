"""Fixed strategy, existing 4+12 development validation samples; no test samples."""
from pathlib import Path
source=Path(__file__).with_name('run_smoke.py').read_text(); exec(source[:source.index('for bits in (2,3,4,5,6,8):')])
assert (OUT/'SUPPLEMENT_COMPLETE').exists()
vae=WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
rows=[]; sources=[]
for name in ('exp_expand32_20261106','exp_a_new48_20261107'):
    root=Path('/mnt/data_hdd/luoxiangyu/dit_pairs')/name
    manifest=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    for m in manifest:
        if m.get('split')=='val' and m['sample_id'] not in [s['sample_id'] for s in sources]: sources.append(m)
assert len(sources)==16
(OUT/'validation_manifest.json').write_text(json.dumps(sources,indent=2))
for s in sources:
    folder=OUT/s['sample_id']; folder.mkdir(exist_ok=True)
    z=torch.load(s['final_latent'],weights_only=True,map_location='cpu').numpy()[None]; rgb,vt,peak,reserved=pixels(z); raw=yuv(rgb)
    original=Path(s['final_latent']).with_name('final_raw.yuv'); assert sha(raw)==sha(original.read_bytes())
    (folder/'reference.yuv').write_bytes(raw)
    (folder/'reference.json').write_text(json.dumps(dict(historical_match=True,latent_sha256=sha(z.tobytes()),source=s,vae_s=vt,peak_allocated=peak,peak_reserved=reserved)))
    configs=[('INT8',dict(bits=8,granularity='channel')),('INT4',dict(bits=4,granularity='channel')),('INT2',dict(bits=2,granularity='channel')),('E1',dict(bits=4,granularity='channel',temporal='integer_delta')),('E2',dict(bits=4,granularity='channel',temporal='closed_loop',refresh=4))]
    direct=None
    for name,cfg in configs:
        path=folder/(name+'.lvc'); sizes,et=timing(lambda:encode(z,path,cfg,dict(algorithm='zstd'),s)); zh,dt=timing(lambda:decode(path))
        if name=='INT4': direct=zh
        if name=='E1': assert np.array_equal(direct,zh)
        rec,vt,_,_=pixels(zh); rr=yuv(rec)
        row=dict(sample=s['sample_id'],split='development_validation',method=name,**sizes,bitrate_kbps=sizes['total_bytes']*8/(17/16)/1000,encode_s=et,decode_s=dt,receiver_vae_s=vt,latent_mse=float(np.mean((z.astype('float64')-zh)**2)),**quality(rgb,rec,raw,rr)); rows.append(row)
    for preset in ('medium','fast'):
        for qp in (18,22,26,28,32,37,42):
            path=folder/f'x264_{preset}_{qp}.264'; args=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i',str(folder/'reference.yuv'),'-c:v','libx264','-preset',preset,'-qp',str(qp),'-g','17','-bf','2','-refs','3','-f','h264',str(path)]
            t=time.perf_counter(); cmd(args); et=time.perf_counter()-t
            rr=cmd(['ffmpeg','-v','error','-i',str(path),'-pix_fmt','yuv420p','-f','rawvideo','pipe:1']); yy=np.frombuffer(raw,dtype='uint8').reshape(17,-1)[:,:832*480].astype('float64'); yy2=np.frombuffer(rr,dtype='uint8').reshape(17,-1)[:,:832*480]
            rows.append(dict(sample=s['sample_id'],split='development_validation',method=f'x264_{preset}_{qp}',total_bytes=path.stat().st_size,bitrate_kbps=path.stat().st_size*8/(17/16)/1000,psnr_y=float(10*np.log10(255**2/np.mean((yy-yy2)**2))),encode_s_single=et))
    (OUT/'validation_rows.json').write_text(json.dumps(rows,indent=2)); print(s['sample_id'],'DONE',flush=True)
(OUT/'VALIDATION_COMPLETE').touch()
