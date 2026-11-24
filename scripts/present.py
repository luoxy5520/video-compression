from pathlib import Path
source=Path(__file__).with_name('run_smoke.py').read_text(); exec(source[:source.index('for bits in (2,3,4,5,6,8):')])
assert (OUT/'SUPPLEMENT_COMPLETE').exists()
vae=WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
rows=json.loads((OUT/'rows.json').read_text())+json.loads((OUT/'supplement.json').read_text()); extras=[]
for s in samples:
    folder=OUT/s['sample']; items=[]
    cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i',str(folder/'reference.rgb'),'-c:v','libx264','-crf','18',str(folder/'reference.mp4')])
    cmd(['ffmpeg','-y','-v','error','-r','16','-i',str(folder/'x264_medium_28.264'),'-c:v','copy',str(folder/'h264_baseline.mp4')])
    for name in ('B2','C2','D2','E1','mixed'):
        rec,_,_,_=pixels(decode(folder/(name+'.lvc'))); recpath=folder/(name+'.yuv'); recpath.write_bytes(yuv(rec))
        args=['ffmpeg','-v','info','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-i',str(folder/'reference.yuv'),'-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-i',str(recpath),'-lavfi','ssim','-f','null','-']
        proc=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True); (folder/(name+'_ssim.log')).write_bytes(proc.stderr)
        import re
        match=re.search(r'All:([0-9.]+)',proc.stderr.decode()); extras.append(dict(sample=s['sample'],method=name,ssim=float(match.group(1))))
    names=['reference','h264_baseline','B2','C2','D2','E1','mixed']; args=['ffmpeg','-y','-v','error']
    for name in names: args+=['-i',str(folder/(name+'.mp4'))]
    filters=[]
    for i,name in enumerate(names):
        row=next((r for r in rows if r['sample']==s['sample'] and r['method']==('x264_medium_28' if name=='h264_baseline' else 'F_mixed' if name=='mixed' else name)),{})
        label={'B2':'INT8 channel AbsMax','C2':'INT4 channel AbsMax','D2':'INT2 channel AbsMax','E1':'INT4 channel integer-delta','mixed':'INT8 sensitive4 + INT4 others'}.get(name,name)
        if row: label+=f" {row.get('bitrate_kbps',0):.1f}kbps PSNR {row.get('psnr_y',0):.2f} enc {row['encode_s']['mean']*1000:.1f}ms"
        filters.append(f"[{i}:v]scale=416:240,drawtext=text='{label}':fontsize=12:fontcolor=white:box=1:boxcolor=black@0.7:x=4:y=4[v{i}]")
    filters.append(''.join(f'[v{i}]' for i in range(7))+'xstack=inputs=7:layout=0_0|416_0|0_240|416_240|0_480|416_480|0_720:fill=black[out]')
    cmd(args+['-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','18',str(folder/'comparison.mp4')])
(OUT/'ssim.json').write_text(json.dumps(extras,indent=2))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for sample in samples:
        fig,ax=plt.subplots()
        for family in ('x264_medium','x264_fast','B','C','D','E'):
            rr=[r for r in rows if r['sample']==sample['sample'] and r['method'].startswith(family)]
            rr.sort(key=lambda r:r['bitrate_kbps']); ax.plot([r['bitrate_kbps'] for r in rr],[r['psnr_y'] for r in rr],'o-',label=family)
        ax.set(xlabel='Bitrate (kbps)',ylabel='PSNR-Y (dB)',title=sample['sample']); ax.legend(); fig.savefig(OUT/(sample['sample']+'_rd.png')); plt.close(fig)
except ImportError as e: (OUT/'plot_NOT_RUN.txt').write_text(str(e))
(OUT/'PRESENT_COMPLETE').touch()
