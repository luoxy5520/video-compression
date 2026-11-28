from common import *
import numpy as np,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=ROOT/'results/E2_1d';rows=[]
for p in out.glob('*/development_step*.json'):rows.extend(json.loads(p.read_text()))
assert rows
simple=[{k:r[k] for k in ['name','step','train_video_count','seed','video','actual_bytes','latent_mse','psnr_y','psnr_rgb','last4_psnr_y']} for r in rows]
csvsave(out/'training_scale_results.csv',simple);csvsave(out/'training_steps_ablation.csv',simple)
uniform=json.loads((E2/'results/intra_pixel.json').read_text());history=json.loads((E2/'results/intra_sweep.json').read_text());oldpixel=json.loads((E2/'results/intra_pixel.json').read_text());h264=json.loads((E2.parent/'results/rows.json').read_text())+json.loads((E2.parent/'results/validation_rows.json').read_text())
matchedr=[];matchedq=[]
def at(v,xkey,ykey,point,logx=False,logy=False):
    v=sorted(v,key=lambda r:r[xkey])
    for a,b in zip(v,v[1:]):
        if a[xkey]<=point<=b[xkey] and a[xkey]!=b[xkey]:
            x1,x2,x=(math.log(a[xkey]),math.log(b[xkey]),math.log(point)) if logx else (a[xkey],b[xkey],point);w=(x-x1)/(x2-x1);y1,y2=(math.log(a[ykey]),math.log(b[ykey])) if logy else (a[ykey],b[ykey]);y=y1+w*(y2-y1);return math.exp(y) if logy else y
for r in rows:
    bs=[dict(actual_bytes=u['actual_bytes'],psnr_y=u['psnr_y']) for u in uniform if u['kind'].startswith('I0') and u['development_path']==r['video']]
    y=at(bs,'actual_bytes','psnr_y',r['actual_bytes'],True)
    if y is not None:matchedr.append(dict(name=r['name'],step=r['step'],video=r['video'],actual_bytes=r['actual_bytes'],psnr_y=r['psnr_y'],uniform_interpolated_psnr_y=y,delta_psnr_y=r['psnr_y']-y,method='within measured Uniform support, linear PSNR in log bytes'))
    size=at(bs,'psnr_y','actual_bytes',r['psnr_y'],False,True)
    if size is not None:matchedq.append(dict(name=r['name'],step=r['step'],video=r['video'],psnr_y=r['psnr_y'],actual_bytes=r['actual_bytes'],uniform_interpolated_bytes=size,saving_fraction=1-r['actual_bytes']/size))
csvsave(out/'matched_rate_comparison.csv',matchedr);csvsave(out/'matched_quality_comparison.csv',matchedq);csvsave(out/'worst_cases.csv',sorted(simple,key=lambda r:r['psnr_y']))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,video in zip(axes,sorted({r['video'] for r in rows})):
    sample=Path(video).parent.name
    for label,series in [('Uniform',[r for r in uniform if r['kind'].startswith('I0') and r['development_path']==video]),('Historical I3 1000',[r for r in oldpixel if r['kind']=='I3' and r['development_path']==video])]:
        series=sorted(series,key=lambda r:r['actual_bytes']);ax.plot([r['actual_bytes']/1000 for r in series],[r['psnr_y'] for r in series],'o-',label=label)
    for seed,style in [(20261127,'s'),(20261128,'x')]:
        series=[r for r in rows if r['video']==video and r['step']==5000 and r['train_video_count']==16 and r['seed']==seed];series=sorted(series,key=lambda r:r['actual_bytes']);ax.plot([r['actual_bytes']/1000 for r in series],[r['psnr_y'] for r in series],style+'--',label=f'Adapted I3 5000 seed{seed}')
    hs=sorted([r for r in h264 if r['sample']==sample and r['method'].startswith('x264_medium_')],key=lambda r:r['total_bytes']);ax.plot([r['total_bytes']/1000 for r in hs],[r['psnr_y'] for r in hs],'.-',label='Historical x264 medium')
    ax.set_title(sample+' (historical development)');ax.set_xlabel('Actual complete kBytes');ax.set_ylabel('PSNR-Y');ax.set_xscale('log');ax.legend(fontsize=7);ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(ROOT/'figures/development_rd_curves.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for folder in out.glob('D*'):
    p=folder/'curve.json'
    if not p.exists():continue
    curve=json.loads(p.read_text());axes[0].plot([r['step'] for r in curve],[r['loss'] for r in curve],label=folder.name);axes[1].plot([r['step'] for r in curve],[r['mse'] for r in curve],label=folder.name)
for ax in axes:ax.set_xlabel('Step');ax.set_yscale('log');ax.legend(fontsize=6);ax.grid(alpha=.3)
axes[0].set_ylabel('Training RD loss (varying batch)');axes[1].set_ylabel('Training latent MSE (varying batch)');fig.tight_layout();fig.savefig(ROOT/'figures/convergence_analysis.png',dpi=160);plt.close(fig)
# Require actual low/middle/high development points before allowing downstream runs.
needed=['D16_f0.1_seed20261127','D16_f1_seed20261127','D16_f10_seed20261127'];points=[r for r in rows if r['step']==5000 and r['name'] in needed]
passed=len(points)==6
if passed:
    for video in {r['video'] for r in points}:
        series=[next(r for r in points if r['video']==video and r['name']==n) for n in needed]
        passed=passed and series[-1]['latent_mse']<series[0]['latent_mse'] and series[-1]['actual_bytes']>series[0]['actual_bytes']
save(out/'gate.json',{'pass':bool(passed),'basis':'three actual development working points, lower high-weight latent MSE and higher bitrate than low-weight; not independent validation','selected_name':'D16_f1_seed20261127','selected_step':10000,'coverage_rows':len(matchedr),'candidate_rows':len(rows),'two_training_seeds_evaluated':True})
print('scale gate',passed,'evaluations',len(rows),'matched rate',len(matchedr))
