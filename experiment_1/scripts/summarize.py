"""Rebuild CSV summaries and figures from measured artifacts only."""
import csv
import json
from pathlib import Path
import numpy as np
from run import ROOT,E0,save
from analyze import table,corr

def ranks(x):
    x=np.asarray(x); _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-counts/2-.5)[inverse]

def interpolate(rows,quality):
    # Measured monotone quality envelope; no extrapolation.
    values=sorted([(v['psnr_y'],v['total_bytes']) for v in rows])
    if len(values)<2 or not values[0][0]<=quality<=values[-1][0]:return None
    q,b=np.array(values).T
    return float(np.exp(np.interp(quality,q,np.log(b))))

def main():
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def output(name):
        plt.tight_layout();plt.savefig(figures/(name+'.png'),dpi=160);plt.close()
    uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
    h264=json.loads((E0/'results/rows.json').read_text())+json.loads((E0/'results/validation_rows.json').read_text())
    e0_validation=json.loads((E0/'results/validation_rows.json').read_text())
    reproduced=[]
    for row in uniform:
        if row['split']!='validation' or row['method'] not in ('Q2','Q4','Q8'):continue
        old=next(v for v in e0_validation if v['sample']==row['sample'] and v['method']=='INT'+row['method'][1:])
        delta=row['psnr_y']-old['psnr_y'];assert abs(delta)<1e-6, 'validation E0 PSNR reproduction mismatch'
        reproduced.append(dict(sample=row['sample'],method=row['method'],old_bytes=old['total_bytes'],new_bytes=row['total_bytes'],byte_delta=row['total_bytes']-old['total_bytes'],old_psnr_y=old['psnr_y'],new_psnr_y=row['psnr_y'],psnr_delta=delta,rate_note='same quantization; metadata fields differ, no algorithmic rate gain claimed'))
    table(ROOT/'results/E1_0/e0_reproduction_validation.csv',reproduced)
    matched=[]
    for row in uniform:
        if row['method']=='Q0':continue
        for preset in ('medium','fast'):
            baseline=[v for v in h264 if v['sample']==row['sample'] and v['method'].startswith(f'x264_{preset}_')]
            estimate=interpolate(baseline,row['psnr_y'])
            matched.append(dict(sample=row['sample'],split=row['split'],method=row['method'],preset=preset,psnr_y=row['psnr_y'],latent_bytes=row['total_bytes'],h264_interpolated_bytes=estimate,latent_over_h264_ratio=row['total_bytes']/estimate if estimate else None,interpolation='log bytes vs PSNR; no extrapolation'))
    table(ROOT/'results/E1_0/quality_matched_h264.csv',matched)
    for split in ('development','validation'):
        names=sorted({v['sample'] for v in uniform if v['split']==split})
        fig,axes=plt.subplots((len(names)+3)//4,min(4,len(names)),figsize=(14,3*((len(names)+3)//4)),squeeze=False)
        for ax,name in zip(axes.ravel(),names):
            vals=[v for v in uniform if v['sample']==name and v['method']!='Q0']
            ax.plot([v['bitrate_kbps'] for v in vals],[v['psnr_y'] for v in vals],'o-',label='LVC1 Q2-Q8')
            base=sorted([v for v in h264 if v['sample']==name and v['method'].startswith('x264_medium_')],key=lambda v:v['bitrate_kbps'])
            ax.plot([v['bitrate_kbps'] for v in base],[v['psnr_y'] for v in base],'x-',label='H.264 medium (E0)');fast=sorted([v for v in h264 if v['sample']==name and v['method'].startswith('x264_fast_')],key=lambda v:v['bitrate_kbps'])
            ax.plot([v['bitrate_kbps'] for v in fast],[v['psnr_y'] for v in fast],'+-',label='H.264 fast (E0)');ax.set_title(name);ax.set_xlabel('kbps');ax.set_ylabel('PSNR-Y dB');ax.legend(fontsize=6)
        output('rd_curve_per_video_'+split)
    for metric,label,file in [('psnr_y','PSNR-Y dB','bitwidth_vs_psnr'),('total_bytes','Complete bitstream bytes','bitwidth_vs_bytes')]:
        for split in ('development','validation'):
            values=[np.mean([v[metric] for v in uniform if v['split']==split and v['method']==f'Q{b}']) for b in range(2,9)]
            plt.plot(range(2,9),values,'o-',label=split)
        plt.xlabel('Uniform bit width');plt.ylabel(label);plt.legend();output(file)
    sensitivity=ROOT/'results/E1_1/sensitivity.json'
    if sensitivity.exists():
        rows=json.loads(sensitivity.read_text());frozen=ROOT/'configs/frozen_channel_ranking.json'
        if frozen.exists():
            rank=json.loads(frozen.read_text())['ranking'];dev_order=np.argsort(rank)
            stability=[];correlations=[]
            stats=list(csv.DictReader((ROOT/'results/E1_1/channel_statistics.csv').open()))
            # Post-hoc analysis only: exact latent error at the perturbed widths.
            # This table never feeds back to the frozen encoder policies.
            import runtime as runtime
            from codec.lvc import decode
            paths={s['sample']:s['latent'] for s in runtime.samples}
            paths.update({s['sample_id']:s['final_latent'] for s in json.loads((E0/'results/validation_manifest.json').read_text())})
            for name in sorted({s['sample'] for s in stats}):
                z=runtime.torch.load(paths[name],weights_only=True,map_location='cpu').numpy()[None]
                for width in (4,6):
                    decoded=decode(ROOT/'results/E1_0'/name/f'Q{width}.lvc')
                    error=np.mean((z.astype('float64')-decoded)**2,axis=(0,2,3,4))
                    for s in stats:
                        if s['sample']==name:s[f'quantization_mse_int{width}']=float(error[int(s['channel'])])
            table(ROOT/'results/E1_1/channel_statistics_augmented.csv',stats)
            names=sorted({v['sample'] for v in rows if v['channel']>=0})
            heat=[];rank_records=[]
            for name in names:
                vals=sorted([v for v in rows if v['sample']==name and v['bits']==4 and v['channel']>=0],key=lambda v:v['channel'])
                if len(vals)!=16:continue
                scores=[v['distortion_increment'] for v in vals];order=np.argsort(scores)[::-1];heat.append(scores)
                sample_ranks=ranks(-np.array(scores))+1
                rank_records.extend(dict(sample=name,split=vals[0]['split'],channel=c,rank=float(sample_ranks[c])) for c in range(16))
                stability.append(dict(sample=name,split=vals[0]['split'],spearman=corr(ranks(scores),-dev_order),top4_overlap=len(set(rank[:4])&set(order[:4]))/4,top8_overlap=len(set(rank[:8])&set(order[:8]))/8))
                for proxy in ('variance','energy','quantization_mse','quantization_mse_int4','quantization_mse_int6','symbol_entropy','temporal_corr'):
                    x=[float(next(v[proxy] for v in stats if v['sample']==name and int(v['channel'])==c)) for c in range(16)]
                    correlations.append(dict(sample=name,split=vals[0]['split'],proxy=proxy,pearson=corr(x,scores),spearman=corr(ranks(x),ranks(scores))))
            table(ROOT/'results/E1_1/channel_rank_stability.csv',stability)
            table(ROOT/'results/E1_1/channel_ranks.csv',rank_records)
            rank_summary=[]
            for c in range(16):
                values=[v['rank'] for v in rank_records if v['channel']==c and v['split']=='validation']
                if values:rank_summary.append(dict(channel=c,n=len(values),mean_rank=float(np.mean(values)),std_rank=float(np.std(values,ddof=1)) if len(values)>1 else None,min_rank=min(values),max_rank=max(values)))
            table(ROOT/'results/E1_1/channel_rank_summary.csv',rank_summary)
            table(ROOT/'results/E1_1/statistics_distortion_correlations.csv',correlations)
            if heat:
                plt.figure(figsize=(10,6));plt.imshow(heat,aspect='auto');plt.colorbar(label='RGB MSE increase vs INT8');plt.yticks(range(len(heat)),names[:len(heat)],fontsize=6);plt.xlabel('Channel');plt.ylabel('Video');output('channel_sensitivity_heatmap')
    blockfile=ROOT/'results/E1_3/block_residual_statistics.csv'
    if blockfile.exists():
        groups={}
        with blockfile.open() as f:
            for row in csv.DictReader(f):
                key=(row['sample'],row['split'],int(row['block_size']),int(row['t']))
                groups.setdefault(key,[]).append(float(row['residual_energy']))
        summaries=[]
        for (sample,split,size,t),energy in groups.items():
            summaries.append(dict(sample=sample,split=split,block_size=size,t=t,blocks=len(energy),mean=float(np.mean(energy)),p25=float(np.percentile(energy,25)),median=float(np.median(energy)),p75=float(np.percentile(energy,75)),p95=float(np.percentile(energy,95)),maximum=float(np.max(energy))))
        table(ROOT/'results/E1_3/block_energy_percentiles.csv',summaries)
        for t in range(1,5):
            data=[v['mean'] for v in summaries if v['block_size']==4 and v['t']==t]
            plt.plot(data,'o-',label=f'{t-1} to {t}')
        plt.ylabel('Mean latent residual energy');plt.xlabel('Video index');plt.legend();output('residual_energy_distribution')
    mixed=ROOT/'results/E1_2/mixed_precision_results.json'
    if mixed.exists():
        rows=json.loads(mixed.read_text());matches=[]
        h264_matches=[]
        for v in rows:
            if v['policy']['method']=='M4_search':continue
            baseline=[u for u in uniform if u['sample']==v['sample'] and u['method']!='Q0']
            base=sorted(baseline,key=lambda u:u['total_bytes']);bs=np.array([u['total_bytes'] for u in base]);ps=np.array([u['psnr_y'] for u in base])
            valid=bs.min()<=v['total_bytes']<=bs.max()
            estimate=float(np.interp(np.log(v['total_bytes']),np.log(bs),ps)) if valid else None
            matches.append(dict(sample=v['sample'],split=v['split'],policy=json.dumps(v['policy']),actual_bytes=v['total_bytes'],psnr_y=v['psnr_y'],uniform_interpolated_psnr=estimate,gain_db=v['psnr_y']-estimate if estimate is not None else None,interpolation='PSNR vs log complete bytes; within measured range only'))
            for preset in ('medium','fast'):
                base=[u for u in h264 if u['sample']==v['sample'] and u['method'].startswith(f'x264_{preset}_')]
                estimate=interpolate(base,v['psnr_y'])
                h264_matches.append(dict(sample=v['sample'],split=v['split'],policy=json.dumps(v['policy']),preset=preset,psnr_y=v['psnr_y'],actual_bytes=v['total_bytes'],h264_interpolated_bytes=estimate,latent_over_h264_ratio=v['total_bytes']/estimate if estimate else None))
        table(ROOT/'results/E1_2/actual_rate_matched_comparison.csv',matches)
        table(ROOT/'results/E1_2/mixed_quality_matched_h264.csv',h264_matches)
    reuse=ROOT/'results/E1_3/skip_probe_results.json'
    if reuse.exists():
        rows=json.loads(reuse.read_text())
        for strategy in ('low','random','high'):
            vals=[v for v in rows if v['block_size']==4 and v['configuration'].get('strategy')==strategy and v['configuration'].get('mode')=='closed_loop']
            ratios=sorted({v['configuration']['ratio'] for v in vals})
            plt.plot(ratios,[np.mean([v['psnr_y'] for v in vals if v['configuration']['ratio']==x]) for x in ratios],'o-',label=strategy)
        plt.xlabel('Replaced block fraction');plt.ylabel('PSNR-Y dB');plt.legend();output('skip_ratio_vs_psnr')
    save(ROOT/'results/summary_status.json',dict(uniform_rows=len(uniform),expected_uniform_rows=144,bd_rate='NOT RUN; per-video interpolation only'))

if __name__=='__main__':main()
