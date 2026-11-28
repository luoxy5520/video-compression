from common import *
import math,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=ROOT/'results/development_pool');p.add_argument('--figure-prefix',default='pool_rd');a=p.parse_args()
out=a.input_dir;rows=json.loads((out/'rows.json').read_text());methods=sorted({r['method'] for r in rows});videos=sorted({r['video'] for r in rows})
def frontier(points):
    result=[];best=-float('inf')
    for r in sorted(points,key=lambda r:(r['actual_bytes'],-r['psnr_y'])):
        if r['psnr_y']>best:result.append(r);best=r['psnr_y']
    return result

def interpolate(points,target,inverse=False):
    for a,b in zip(points,points[1:]):
        x1,x2=(a['psnr_y'],b['psnr_y']) if inverse else (math.log(a['actual_bytes']),math.log(b['actual_bytes']));x=target if inverse else math.log(target)
        if x1<=x<=x2 and x2>x1:
            w=(x-x1)/(x2-x1)
            return math.exp(math.log(a['actual_bytes'])+w*math.log(b['actual_bytes']/a['actual_bytes'])) if inverse else a['psnr_y']+w*(b['psnr_y']-a['psnr_y'])
    return None
summary=[];matched=[];matchedq=[];dominance=[];controls=[];h264matched=[];h264matchedq=[]
for method in methods:
    x=[r for r in rows if r['method']==method];summary.append(dict(method=method,videos=len(x),mean_actual_bytes=float(np.mean([r['actual_bytes'] for r in x])),mean_psnr_y=float(np.mean([r['psnr_y'] for r in x])),mean_psnr_rgb=float(np.mean([r['psnr_rgb'] for r in x])),mean_SSIM_RGB=float(np.mean([r['SSIM_RGB'] for r in x])),mean_last4_psnr_y=float(np.mean([r['last4_psnr_y'] for r in x])),worst_psnr_y=min(r['psnr_y'] for r in x)))
for video in videos:
    uni=frontier([r for r in rows if r['video']==video and r['method'].startswith('Uniform_INT')]);selected={r['method']:r for r in rows if r['video']==video}
    h264=frontier([r for r in rows if r['video']==video and r['method'].startswith('H264_medium_QP')])
    for r in [r for r in rows if r['video']==video and not r['method'].startswith(('Uniform_INT','H264_medium_QP'))]:
        y=interpolate(h264,r['actual_bytes']);size=interpolate(h264,r['psnr_y'],True)
        if y is not None:h264matched.append(dict(method=r['method'],video=video,actual_bytes=r['actual_bytes'],psnr_y=r['psnr_y'],h264_interpolated_psnr_y=y,delta_psnr_y=r['psnr_y']-y))
        if size is not None:h264matchedq.append(dict(method=r['method'],video=video,psnr_y=r['psnr_y'],actual_bytes=r['actual_bytes'],h264_interpolated_bytes=size,saving_fraction=1-r['actual_bytes']/size))
    for r in [r for r in rows if r['video']==video and not r['method'].startswith('Uniform_INT')]:
        y=interpolate(uni,r['actual_bytes']);size=interpolate(uni,r['psnr_y'],True)
        if y is not None:matched.append(dict(method=r['method'],video=video,actual_bytes=r['actual_bytes'],psnr_y=r['psnr_y'],uniform_interpolated_psnr_y=y,delta_psnr_y=r['psnr_y']-y))
        if size is not None:matchedq.append(dict(method=r['method'],video=video,psnr_y=r['psnr_y'],actual_bytes=r['actual_bytes'],uniform_interpolated_bytes=size,saving_fraction=1-r['actual_bytes']/size))
        if y is None:
            for u in uni:
                if r['actual_bytes']<u['actual_bytes'] and r['psnr_y']>u['psnr_y']:dominance.append(dict(method=r['method'],video=video,codec_bytes=r['actual_bytes'],codec_psnr_y=r['psnr_y'],uniform_method=u['method'],uniform_bytes=u['actual_bytes'],uniform_psnr_y=u['psnr_y']))
    if all(m in selected for m in ('TP0','TP1','TP2','TP2Z')):controls.append(dict(video=video,TP0_bytes=selected['TP0']['actual_bytes'],TP1_bytes=selected['TP1']['actual_bytes'],TP2_bytes=selected['TP2']['actual_bytes'],zero_history_same_head_bytes=selected['TP2Z']['actual_bytes'],TP2_saving_vs_TP0=1-selected['TP2']['actual_bytes']/selected['TP0']['actual_bytes'],TP2_independent_history_saving=1-selected['TP2']['actual_bytes']/selected['TP2Z']['actual_bytes']))
    fig,ax=plt.subplots(figsize=(8,5))
    groups=[('Uniform',[m for m in methods if m.startswith('Uniform_INT')]),('H.264 medium',[m for m in methods if m.startswith('H264_medium_QP')]),('Historical I3',[m for m in methods if m.startswith('historical_I3')]),('Trained I3',['trained_I3_f0.1','trained_I3_f1','trained_I3_f10']),('V2-1000',['V2_low_D16','V2','V2_high_D16'])]
    for label,names in groups:
        ps=sorted([r for r in rows if r['video']==video and r['method'] in names],key=lambda r:r['actual_bytes'])
        ax.plot([r['actual_bytes']/1000 for r in ps],[r['psnr_y'] for r in ps],'o-',label=label)
    extra=[m for m in methods if m.startswith(('P2_long','P3_long','D42_temporal'))]
    for method in ['V1','V2_mid_3000','TP2','P2','P3']+extra:
        if method in selected:r=selected[method];ax.scatter([r['actual_bytes']/1000],[r['psnr_y']],label=method)
    ax.set_xscale('log');ax.set_xlabel('Actual complete kBytes');ax.set_ylabel('PSNR-Y');ax.set_title(Path(video).parent.parent.name+' / '+Path(video).parent.name+' / historical development');ax.grid(alpha=.3);ax.legend(fontsize=6,ncol=2 if extra else 1);fig.tight_layout();fig.savefig(ROOT/'figures'/f'{a.figure_prefix}_{videos.index(video):02d}_{Path(video).parent.name}.png',dpi=160);plt.close(fig)
coverage=[]
for method in methods:
    mr=[r for r in matched if r['method']==method];mq=[r for r in matchedq if r['method']==method];coverage.append(dict(method=method,evaluated_videos=sum(r['method']==method for r in rows),historical_pool_videos=len(videos),matched_rate_videos=len(mr),matched_quality_videos=len(mq),mean_matched_rate_delta_db=float(np.mean([r['delta_psnr_y'] for r in mr])) if mr else 'NOT COMPARABLE',mean_matched_quality_saving=float(np.mean([r['saving_fraction'] for r in mq])) if mq else 'NOT COMPARABLE'))
h264coverage=[]
for method in methods:
    if method.startswith(('Uniform_INT','H264_medium_QP')):continue
    mr=[r for r in h264matched if r['method']==method];mq=[r for r in h264matchedq if r['method']==method]
    h264coverage.append(dict(method=method,evaluated_videos=sum(r['method']==method for r in rows),historical_pool_videos=len(videos),matched_rate_videos=len(mr),matched_quality_videos=len(mq),mean_matched_rate_delta_db=float(np.mean([r['delta_psnr_y'] for r in mr])) if mr else 'NOT COMPARABLE',mean_matched_quality_saving=float(np.mean([r['saving_fraction'] for r in mq])) if mq else 'NOT COMPARABLE'))
for name,data in [('summary.csv',summary),('coverage.csv',coverage),('matched_rate_comparison.csv',matched),('matched_quality_comparison.csv',matchedq),('h264_coverage.csv',h264coverage),('h264_matched_rate_comparison.csv',h264matched),('h264_matched_quality_comparison.csv',h264matchedq),('temporal_controlled_comparison.csv',controls),('out_of_support_measured_dominance.csv',dominance)]:
    if data:csvsave(out/name,data)
rate_rows=[dict(method=r['method'],video=r['video'],**s) for r in rows for s in r.get('rate_diagnostics',[])]
if rate_rows:csvsave(out/'estimated_vs_actual_temporal_rate.csv',rate_rows)
csvsave(out/'worst_cases.csv',[{k:r[k] for k in ['method','video','actual_bytes','psnr_y','psnr_rgb','SSIM_RGB','last4_psnr_y']} for r in sorted(rows,key=lambda r:r['psnr_y'])])
save(out/'summary.json',dict(summary=summary,coverage=coverage,h264_coverage=h264coverage,videos=len(videos),rows=len(rows),independent_validation=False,sealed_test='NOT OPENED',interpolation='measured Pareto support only; no extrapolation; matched subsets can differ between methods',BD_rate='NOT RUN: sparse/nonmonotonic learned points; no validated BD integration'))
print('pool summarized',len(videos),len(rows))
