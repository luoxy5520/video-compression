"""Measured before/after changes; no equal-rate assumption or extrapolation."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
old={r['video']:r for r in json.loads((ROOT/'results/development_pool/rows.json').read_text()) if r['method']=='P2'}
new=[r for r in json.loads((ROOT/'results/extension_evaluation/gpu0/rows.json').read_text()) if r['method']=='P2_long10000_P2']
assert len(new)==18
rows=[]
for r in new:
    b=old[r['video']];rows.append(dict(video=r['video'],old_bytes=b['actual_bytes'],new_bytes=r['actual_bytes'],bytes_change_fraction=r['actual_bytes']/b['actual_bytes']-1,old_psnr_y=b['psnr_y'],new_psnr_y=r['psnr_y'],psnr_y_change=r['psnr_y']-b['psnr_y'],measured_dominates=r['actual_bytes']<=b['actual_bytes'] and r['psnr_y']>=b['psnr_y']))
csvsave(ROOT/'results/extension_interim_gpu0/P2_step_paired.csv',rows)
fig,ax=plt.subplots(figsize=(8,5))
for flag,label,color in [(True,'Bytes no larger, PSNR no lower','tab:green'),(False,'Measured tradeoff','tab:orange')]:
    subset=[r for r in rows if r['measured_dominates']==flag]
    ax.scatter([r['bytes_change_fraction']*100 for r in subset],[r['psnr_y_change'] for r in subset],label=f'{label} ({len(subset)}/18)',color=color)
for r in rows:
    if not r['measured_dominates']:ax.annotate(Path(r['video']).parent.name,(r['bytes_change_fraction']*100,r['psnr_y_change']),xytext=(4,5),textcoords='offset points',fontsize=8)
ax.axhline(0,color='grey',linewidth=.8);ax.axvline(0,color='grey',linewidth=.8)
ax.set_xlabel('Actual complete Bytes change (%)');ax.set_ylabel('PSNR-Y change (dB)');ax.set_title('P2: 1500 to10000 steps /18 historical development videos');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.margins(.16,.2)
fig.tight_layout();fig.savefig(ROOT/'figures/P2_continuation_paired.png',dpi=160);plt.close(fig)
