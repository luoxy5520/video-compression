import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'tools_plot'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=json.loads((ROOT/'results/rows.json').read_text())
for sample in sorted({r['sample'] for r in rows}):
    rr=[r for r in rows if r['sample']==sample]
    for xkey,ykey,label in [('bitrate_kbps','psnr_y','rd'),('encode_ms','psnr_y','encoding_quality'),('bitrate_kbps','encode_ms','encoding_rate')]:
        fig,ax=plt.subplots(figsize=(7,5))
        for family in ('x264_medium','x264_fast','Q8','Q4','Q2','E2'):
            points=[r for r in rr if r['method'].startswith(family)]
            for r in points: r['encode_ms']=r['encode_s']['mean']*1000
            points.sort(key=lambda r:r[xkey])
            ax.plot([r[xkey] for r in points],[r[ykey] for r in points],'o-',label=family)
        ax.set(xlabel=xkey,ylabel=ykey,title=sample); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'results'/f'{sample}_{label}.png'); plt.close(fig)
ssim=json.loads((ROOT/'results/ssim.json').read_text())
for sample in sorted({r['sample'] for r in ssim}):
    fig,ax=plt.subplots(figsize=(7,5))
    for r in ssim:
        if r['sample']!=sample: continue
        rate=next((x['bitrate_kbps'] for x in rows if x['sample']==sample and x['method']==r['method']),None)
        if rate is None: continue
        ax.scatter(rate,r['ssim']); ax.annotate(r['method'],(rate,r['ssim']))
    ax.set(xlabel='Bitrate (kbps)',ylabel='SSIM',title=sample); fig.tight_layout(); fig.savefig(ROOT/'results'/f'{sample}_ssim.png'); plt.close(fig)
bench=ROOT/'results/benchmark.json'
if bench.exists():
    b=json.loads(bench.read_text()); fig,ax=plt.subplots(figsize=(7,5))
    for sample in sorted({r['sample'] for r in b}):
        for method,lookup in [('x264_medium_28','x264_medium_28'),('INT8','B2'),('INT4','C2')]:
            rr=[r for r in b if r['sample']==sample and r['method']==method]; q=next(r for r in rows if r['sample']==sample and r['method']==lookup)
            ax.scatter(sum(r['post_dit_to_receiver_pixels_s'] for r in rr)/len(rr)*1000,q['psnr_y']); ax.annotate(sample+' '+method,(sum(r['post_dit_to_receiver_pixels_s'] for r in rr)/len(rr)*1000,q['psnr_y']),fontsize=7)
    ax.set(xlabel='Post-DiT through receiver pixels (ms)',ylabel='PSNR-Y (dB)',title='Different qualities; no Prompt stages'); fig.tight_layout(); fig.savefig(ROOT/'results/post_dit_receiver_quality.png'); plt.close(fig)
validation=ROOT/'results/validation_rows.json'
if validation.exists():
    import statistics
    v=json.loads(validation.read_text()); fig,ax=plt.subplots(figsize=(7,5))
    for family in ('x264_medium','x264_fast'):
        points=[]
        for qp in (18,22,26,28,32,37,42):
            rr=[r for r in v if r['method']==f'{family}_{qp}']; points.append((statistics.mean(r['bitrate_kbps'] for r in rr),statistics.mean(r['psnr_y'] for r in rr)))
        points.sort(); ax.plot([p[0] for p in points],[p[1] for p in points],'o-',label=family)
    for method in ('INT8','INT4','INT2','E1','E2'):
        rr=[r for r in v if r['method']==method]; x=statistics.mean(r['bitrate_kbps'] for r in rr); y=statistics.mean(r['psnr_y'] for r in rr); ax.scatter(x,y,label=method)
    ax.set(xlabel='Mean bitrate (kbps)',ylabel='Mean per-video PSNR-Y (dB)',title='16 development-validation samples'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(ROOT/'results/validation_rd.png'); plt.close(fig)
