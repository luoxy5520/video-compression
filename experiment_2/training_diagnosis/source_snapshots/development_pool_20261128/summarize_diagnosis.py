from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=ROOT/'results/E2_1b';rows=[];shape=[]
fig,axes=plt.subplots(1,2,figsize=(11,4))
for name in ('R1','R2','R3','R3_identity'):
    p=b/name/'pixel_results.json'
    if not p.exists():continue
    results=json.loads(p.read_text());own=[r for r in results if r['variant']==name]
    for x in own:rows.append({k:x[k] for k in ['variant','step','mse','mae','relative_l2','psnr_y','psnr_rgb','last4_psnr_y']})
    steps=[r['step'] for r in own];axes[0].plot(steps,[r['mse'] for r in own],'o-',label=name);axes[1].plot(steps,[r['psnr_y'] for r in own],'o-',label=name)
    for p in (b/name).glob('step*.json'):
        s=json.loads(p.read_text());shape.append({k:s[k] for k in ['variant','step','input_shape','coding_shape','bottleneck_element_ratio']})
    if name=='R1':
        for r in results[:3]:axes[1].axhline(r['psnr_y'],ls=':',label=r['variant'])
axes[0].set_yscale('log');axes[0].set_ylabel('Continuous latent MSE');axes[1].set_ylabel('PSNR-Y (dB)')
for ax in axes:ax.set_xlabel('Training step (identity witness at 0)');ax.grid(alpha=.3);ax.legend(fontsize=7)
fig.suptitle('No quantization / no rate: reconstruction capacity only');fig.tight_layout();fig.savefig(ROOT/'figures/reconstruction_training_curve.png',dpi=160);plt.close(fig)
csvsave(b/'reconstruction_only_results.csv',rows);csvsave(b/'architecture_capacity_ablation.csv',rows);save(b/'coding_latent_shape_analysis.json',shape)
(b/'E2_1b_report.md').write_text('# E2.1b\n\nR1 random initialization: input499200 elements, coding249600 (ratio.5). R2 adds32→64 coding channels (ratio1), otherwise same transforms. R3 cancels the sole spatial downsample (ratio2); no large CNN replacement. Randomly trained R1/R2/R3 improve but remain below INT6 in this diagnostic budget. R3 also admits an analytical identity initialization because GELU(x)-GELU(-x)=x; tested on saved full video and random tensor (<2e-6 max error). Identity is a capacity witness and a separate new initialization, not compression performance or convergence evidence. It is used to remove reconstruction-optimization conditioning as a confound in O1/O2. Exact measured CSV and full-sequence VAE curves accompany this report.\n')
c=ROOT/'results/E2_1c';p=c/'single_video_overfit.csv'
if p.exists():
    import csv
    points=list(csv.DictReader(p.open()));fig,ax=plt.subplots(figsize=(6,4));ax.plot([float(x['actual_bytes'])/1000 for x in points],[float(x['psnr_y']) for x in points],'o-');ax.set_xlabel('Actual complete kBytes');ax.set_ylabel('PSNR-Y');ax.set_title('Single training video, not generalization');fig.tight_layout();fig.savefig(ROOT/'figures/overfit_rd_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for folder in sorted(c.glob('O2*')):
        if not (folder/'curve.json').exists():continue
        curve=json.loads((folder/'curve.json').read_text());axes[0].plot([r['step'] for r in curve],[r['rate'] for r in curve],label=folder.name.split('_f')[-1]);axes[1].plot([r['step'] for r in curve],[r['mse'] for r in curve],label=folder.name.split('_f')[-1])
    axes[0].set_ylabel('Estimated bits/input latent element');axes[1].set_ylabel('Training latent MSE')
    for ax in axes:ax.set_xlabel('Step');ax.legend(title='lambda factor');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(ROOT/'figures/overfit_loss_curves.png',dpi=160);plt.close(fig)
print('diagnosis summary written')
