from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=ROOT/'results/E2_1e';rows=[];baseline=json.loads((ROOT/'results/E2_1d/D16_f1_seed20261127/development_step10000.json').read_text())
for r in baseline:rows.append(dict(method='V0',steps=10000,video=r['video'],actual_bytes=r['actual_bytes'],latent_mse=r['latent_mse'],psnr_y=r['psnr_y'],psnr_rgb=r['psnr_rgb'],last4_psnr_y=r['last4_psnr_y']))
for mode in ('V1','V2'):
    for step in (100,1000):
        p=out/f'{mode}_D16'/f'development_step{step}.json'
        if not p.exists():continue
        for r in json.loads(p.read_text()):rows.append(dict(method=mode,steps=step,video=r['video'],actual_bytes=r['actual_bytes'],latent_mse=r['latent_mse'],psnr_y=r['psnr_y'],psnr_rgb=r['psnr_rgb'],last4_psnr_y=r['last4_psnr_y']))
csvsave(out/'pixel_loss_ablation.csv',rows)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,video in zip(axes,sorted({r['video'] for r in rows})):
    for method in ('V0','V1','V2'):
        x=[r for r in rows if r['video']==video and r['method']==method];ax.plot([r['actual_bytes']/1000 for r in x],[r['psnr_y'] for r in x],'o-',label=method)
    ax.set_title(Path(video).parent.name);ax.set_xlabel('Actual complete kBytes');ax.set_ylabel('PSNR-Y');ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(ROOT/'figures/vae_aware_rd_curve.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for method in ('V0','V1','V2'):
    x=[r for r in rows if r['method']==method];ax.scatter([r['latent_mse'] for r in x],[r['psnr_y'] for r in x],label=method)
ax.set_xlabel('Latent MSE');ax.set_ylabel('PSNR-Y');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/latent_vs_pixel_distortion.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for mode in ('V1','V2'):
    p=out/f'{mode}_D16/curve.json'
    if not p.exists():continue
    curve=json.loads(p.read_text())
    for ax,key in zip(axes,('rate','latent_mse','pixel_mse')):ax.plot([r['step'] for r in curve],[r[key] for r in curve],label=mode);ax.set_ylabel(key);ax.set_xlabel('Step');ax.legend();ax.set_yscale('log')
fig.tight_layout();fig.savefig(ROOT/'figures/pixel_loss_training_curves.png',dpi=160);plt.close(fig)
checks=[]
for p in out.glob('vae_gradient*.json'):
    c=json.loads(p.read_text());checks.append(dict(precision=c['precision'],checkpoint_chunks=c.get('checkpoint_chunks',False),status=c['status'],peak_gpu_bytes=c['peak_gpu_bytes'],forward_seconds=c.get('vae_forward_seconds','NOT RUN'),backward_seconds=c.get('backward_seconds','NOT RUN'),vae_unchanged=c['vae_tensor_sha256_before']==c['vae_tensor_sha256_after']))
csvsave(out/'vae_training_memory.csv',checks)
save(out/'vae_gradient_check.json',json.loads((out/'vae_gradient_fp32_checkpoint.json').read_text()))
# Before/after points are not silently called equal-rate successes.
comparisons=[]
for r in rows:
    if r['method']=='V0':continue
    b=next(x for x in rows if x['method']=='V0' and x['video']==r['video']);comparisons.append(dict(**r,bytes_change_fraction=r['actual_bytes']/b['actual_bytes']-1,psnr_y_change=r['psnr_y']-b['psnr_y'],measured_dominates_V0=bool(r['actual_bytes']<=b['actual_bytes'] and r['psnr_y']>=b['psnr_y']),scope='direct measured before/after; not equal-rate interpolation'))
save(out/'pixel_before_after.json',comparisons);print('pixel rows',len(rows))
