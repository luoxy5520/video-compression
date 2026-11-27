"""Summarize measured development rows; interpolations explicitly diagnostic."""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
pixel=json.loads((R/'results/intra_pixel.json').read_text());sweep=json.loads((R/'results/intra_sweep.json').read_text());sym=json.loads((R/'results/symbol_entropy.json').read_text())
assert len(pixel)==30 and len(sweep)==16 and len(sym)==8, 'incomplete runs'
comparisons=[]
for row in pixel:
    if row['kind'].startswith('I0'):continue
    base=sorted([x for x in pixel if x['kind'].startswith('I0') and x['development_path']==row['development_path']],key=lambda x:x['actual_bytes'])
    for a,b in zip(base,base[1:]):
        if a['actual_bytes']<=row['actual_bytes']<=b['actual_bytes']:
            w=math.log(row['actual_bytes']/a['actual_bytes'])/math.log(b['actual_bytes']/a['actual_bytes']);ref=a['psnr_y']+w*(b['psnr_y']-a['psnr_y'])
            comparisons.append(dict(kind=row['kind'],lambda_z=row['lambda_z'],video=row['development_path'],actual_bytes=row['actual_bytes'],psnr_y=row['psnr_y'],uniform_lograte_interpolated_psnr_y=ref,diagnostic_delta_db=row['psnr_y']-ref,scope='sparse Uniform-curve interpolation, NOT measured equal-rate comparison'))
            break
(R/'results/development_comparison.json').write_text(json.dumps(comparisons,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
paths=list(dict.fromkeys(x['development_path'] for x in pixel))
for ax,path in zip(axes,paths):
    base=sorted([x for x in pixel if x['kind'].startswith('I0') and x['development_path']==path],key=lambda x:x['actual_bytes']);ax.plot([x['actual_bytes']/1000 for x in base],[x['psnr_y'] for x in base],'o-',label='Uniform + Zstd')
    for kind in ('I2','I3'):
        rows=sorted([x for x in pixel if x['kind']==kind and x['development_path']==path],key=lambda x:x['actual_bytes']);ax.plot([x['actual_bytes']/1000 for x in rows],[x['psnr_y'] for x in rows],'o--',label=kind+' short train')
    ax.set_xscale('log');ax.set_xlabel('Complete bitstream (kBytes)');ax.set_ylabel('PSNR-Y (dB)');ax.set_title(Path(path).parent.name+' (historical development)');ax.grid(alpha=.3);ax.legend()
fig.suptitle('16 training videos; 1000 steps; development-only measured points');fig.tight_layout();fig.savefig(R/'figures/intra_development_rd.png',dpi=160);plt.close(fig)
text=['# E2 resumed experiment report','', 'The SSH interruption has been resolved. I2/I3 smoke and repeat passed. Eight short intra runs and fixed-symbol I1 comparison completed. This is a development-stage report, not a final independent evaluation.','', '## Fixed-symbol I1 versus I0','', '16 original train videos; two separate historical development groups. Identical quantized symbols and bit-exact latent reconstruction. One independent-process decode per bitwidth passed. Full files include metadata; shared weights reported separately.','', '| Bits | Mean I0 Bytes | Mean I1 Bytes | I1 byte change |','|---|---:|---:|---:|']
for bits in (4,5,6,8):
    rows=[x for x in sym if x['bits']==bits];a=sum(x['I0_bytes'] for x in rows)/2;b=sum(x['I1_bytes'] for x in rows)/2;text.append(f'| {bits} | {a:.1f} | {b:.1f} | {(b/a-1)*100:+.2f}% |')
text+=['','I1 uses a simple learned per-channel zero-mean Gaussian, not a contextual model. INT6 savings are small; other tested widths regress. Spatial/temporal dependencies retained by Zstd are not represented by this independent Gaussian.','', '## Short intra runs','', '16 train videos, two group-disjoint historical development videos, 1000 steps per model, four weights per architecture, separate entropy auxiliary optimizer. Native frozen Wan VAE decoded recovered latents. No VAE at encoder.','', '| Video | Model | Lambda | Actual Bytes | PSNR-Y | PSNR-RGB |','|---|---|---:|---:|---:|---:|']
for x in pixel:text.append(f"| {Path(x['development_path']).parent.name} | {x['kind']} | {x.get('lambda_z','—')} | {x['actual_bytes']} | {x['psnr_y']:.3f} | {x['psnr_rgb']:.3f} |")
text+=['','Curves span substantially different rates. `development_comparison.json` contains per-video log-rate interpolation only within measured Uniform support; these sparse interpolations do not establish the preset matched-rate gain threshold. No extrapolation is used. Four rate weights need tuning: these short runs do not produce a monotonic RD curve.','', 'I2/I3 real-byte main/hyper/metadata breakdown, resource use, model sizes and SHA256 are in intra_sweep.json; full training curves are retained. At least one slice per checkpoint was decoded in an independent process for the first development video. All slices were decoded from actual files and compared to forward reconstruction during training evaluation.','', '## Gates and remaining work','', '- Real-stream gate: PASS for I2/I3, repeated; odd-size/checkpoint/corruption regression PASS.','- Independent validation: NOT RUN; original validation remains development; sealed test unopened.','- I4 and P0–P4 temporal experiments: NOT RUN. Need a selected, converged intra operating point and finer matched-rate evaluation before increasing architecture complexity.','- Pixel-domain training: NOT RUN; VAE is evaluation-only in this round.','- H.264 matched-quality comparison and preset 0.5dB/10% target: NOT ESTABLISHED. Historical mean is not used as a per-video comparator.','- Formal five-repeat timings, acceleration, FLOPs, RSS, SSIM/LPIPS/DISTS: NOT RUN. VAE diagnostic times in smoke are not formal latency.','- Full GLVC training-strategy reading and ELIC source verification remain incomplete due source access failures, recorded in literature_mapping.md.','', 'No final deployment architecture is selected. No generator, original VAE weights, denoising steps, E0/E1 code or sealed test data were changed.']
(R/'reports/resumed_report.md').write_text('\n'.join(text)+'\n');(R/'final_report.md').write_text('\n'.join(text)+'\n')
print(json.dumps(dict(pixel_rows=len(pixel),train_runs=8,diagnostic_comparisons=len(comparisons))))
