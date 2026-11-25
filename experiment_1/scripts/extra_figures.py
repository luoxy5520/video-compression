"""Additional required plots and CSV views, derived without retuning policies."""
import csv
import json
import shutil
import struct
import numpy as np
from run import ROOT,E0,save
from analyze import table,corr
from summarize import ranks

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    def fig(name):plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=160);plt.close()
    uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
    q=[v for v in uniform if v['method']!='Q0']
    x=[np.mean([v['bitrate_kbps'] for v in q if v['method']==f'Q{b}' and v['split']=='validation']) for b in range(2,9)]
    y=[np.mean([v['psnr_y'] for v in q if v['method']==f'Q{b}' and v['split']=='validation']) for b in range(2,9)]
    plt.plot(x,y,'o-',label='Uniform Q2-Q8')
    baseline=json.loads((E0/'results/validation_rows.json').read_text())
    for preset in ('medium','fast'):
        rates=[];qualities=[]
        for qp in (42,37,32,28,26,22,18):
            vals=[v for v in baseline if v['method']==f'x264_{preset}_{qp}']
            assert len(vals)==16
            rates.append(np.mean([v['bitrate_kbps'] for v in vals]));qualities.append(np.mean([v['psnr_y'] for v in vals]))
        plt.plot(rates,qualities,'x-',label='H.264 '+preset)
    plt.xlabel('Mean actual kbps (16 validation videos)');plt.ylabel('Mean PSNR-Y dB');plt.legend();fig('rd_curve_summary')
    shutil.copyfile(out/'rd_curve_per_video_validation.png',out/'rd_curve_per_video.png')
    header=[np.mean([v['header_bytes'] for v in q if v['method']==f'Q{b}']) for b in range(2,9)]
    payload=[np.mean([v['payload_bytes'] for v in q if v['method']==f'Q{b}']) for b in range(2,9)]
    plt.bar(range(2,9),payload,label='Compressed payload');plt.bar(range(2,9),header,bottom=payload,label='Header incl. quantization');plt.xlabel('Bit width');plt.ylabel('Complete bytes');plt.legend();fig('bitstream_breakdown')
    sensitivity=json.loads((ROOT/'results/E1_1/sensitivity.json').read_text())
    stats=list(csv.DictReader((ROOT/'results/E1_1/channel_statistics_augmented.csv').open()))
    frozen=json.loads((ROOT/'configs/frozen_channel_ranking.json').read_text())
    rank=frozen['ranking'];plt.bar(range(16),[frozen['scores'][c] for c in rank]);plt.xticks(range(16),rank);plt.xlabel('Channel, development sensitivity order');plt.ylabel('Mean RGB MSE increment');fig('channel_importance_ranking')
    for proxy,file in [('energy','energy_vs_distortion'),('quantization_mse_int4','quantization_error_vs_distortion')]:
        for split in ('development','validation'):
            values=[v for v in sensitivity if v['split']==split and v['bits']==4 and v['channel']>=0]
            x=[float(next(s[proxy] for s in stats if s['sample']==v['sample'] and int(s['channel'])==v['channel'])) for v in values]
            plt.scatter(x,[v['distortion_increment'] for v in values],s=8,alpha=.5,label=split)
        plt.xlabel(proxy);plt.ylabel('RGB MSE increase, channel INT4 perturbation');plt.legend();fig(file)
    table(ROOT/'results/E1_3/channel_temporal_correlation.csv',[{k:s[k] for k in ('sample','split','channel','temporal_corr')} for s in stats])
    temporal=list(csv.DictReader((ROOT/'results/E1_3/temporal_similarity.csv').open()))
    figobj,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name in zip(axes,('sample_0000','sample_1004','sample_1032')):
        matrix=np.eye(5)
        for v in temporal:
            if v['sample']==name:matrix[int(v['t0']),int(v['t1'])]=matrix[int(v['t1']),int(v['t0'])]=float(v['pearson'])
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(name);ax.set_xlabel('Latent time slice');ax.set_ylabel('Latent time slice')
    figobj.colorbar(im,ax=axes.ravel().tolist(),label='Pearson correlation');figobj.savefig(out/'temporal_correlation_matrix.png',dpi=160);plt.close()
    maps={}
    with (ROOT/'results/E1_3/block_residual_statistics.csv').open() as f:
        for v in csv.DictReader(f):
            if v['sample'] in ('sample_0000','sample_1004','sample_1032') and int(v['block_size'])==4:
                maps.setdefault((v['sample'],int(v['t'])),np.zeros((15,26)))[int(v['y'])//4,int(v['x'])//4]=float(v['residual_energy'])
    figobj,axes=plt.subplots(3,4,figsize=(14,8))
    for i,name in enumerate(('sample_0000','sample_1004','sample_1032')):
        for t in range(1,5):
            ax=axes[i,t-1];im=ax.imshow(maps[name,t],cmap='magma',vmin=0,vmax=max(float(maps[name,u].max()) for u in range(1,5)));ax.set_title(f'{name} {t-1}->{t}',fontsize=8);figobj.colorbar(im,ax=ax)
    fig('latent_residual_heatmaps')
    mixed=json.loads((ROOT/'results/E1_2/mixed_precision_results.json').read_text())
    allocations=[];breakdowns=[]
    def split_stream(blob):
        n=struct.unpack('<I',blob[4:8])[0];h=json.loads(blob[8:8+n]);payload=blob[8+n:]
        # Count serialized scale and quantization values as part of header, not extra bytes.
        quant=sum(len(json.dumps(h[k],separators=(',',':')).encode()) for k in ('scale','quantization','channel_bits') if k in h)
        aux=sum(len(json.dumps(h[k],separators=(',',':')).encode()) for k in ('lengths','shape','video','original_dtype','layout','byte_order','zero_point','shuffle_itemsize','raw_payload_bytes','original_sha256','compression','payload_sha256','payload_bytes') if k in h)
        if not h.get('mixed'):return n+8,quant,aux,len(payload)
        offset=0;header=n+8;actual=0
        for length in h['lengths']:
            ch,cq,ca,cp=split_stream(payload[offset:offset+length]);header+=ch;quant+=cq;aux+=ca;actual+=cp;offset+=length
        assert offset==len(payload)
        return header,quant,aux,actual
    seen=set()
    for v in mixed:
        if v['policy']['method']=='M4_search':continue
        for c,b in enumerate(v['channel_bits']):allocations.append(dict(sample=v['sample'],split=v['split'],policy=json.dumps(v['policy']),channel=c,bits=b))
        path=ROOT/'results/E1_2'/v['sample']/('bits_'+''.join(map(str,v['channel_bits']))+'.lvc')
        if path in seen:continue
        seen.add(path);header,quant,aux,payload=split_stream(path.read_bytes())
        assert header+payload==path.stat().st_size
        breakdowns.append(dict(sample=v['sample'],stream=str(path.relative_to(ROOT)),total_bytes=path.stat().st_size,header_bytes_including_quant=header,quantization_json_value_bytes=quant,auxiliary_json_value_bytes=aux,other_header_bytes=header-quant-aux,compressed_payload_bytes=payload,definition='quantization/auxiliary JSON values counted separately; keys/delimiters in other header; nested headers recursively included'))
    table(ROOT/'results/E1_2/bit_allocation_map.csv',allocations)
    for v in uniform:
        path=ROOT/'results/E1_0'/v['sample']/(v['method']+'.lvc');header,quant,aux,payload=split_stream(path.read_bytes())
        breakdowns.append(dict(sample=v['sample'],stream=str(path.relative_to(ROOT)),total_bytes=path.stat().st_size,header_bytes_including_quant=header,quantization_json_value_bytes=quant,auxiliary_json_value_bytes=aux,other_header_bytes=header-quant-aux,compressed_payload_bytes=payload,definition='quantization/auxiliary JSON values counted separately; keys/delimiters in other header; nested headers recursively included'))
    table(ROOT/'results/actual_bitstream_breakdown.csv',breakdowns)
    figobj,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,('sample_0000','sample_1004')):
        for family in ('M1','M2','M2_three','M3','M4_approximate_oracle'):
            vals=[v for v in mixed if v['sample']==name and v['policy']['method']==family];ax.scatter([v['bitrate_kbps'] for v in vals],[v['psnr_y'] for v in vals],s=9,alpha=.5,label=family)
        vals=[v for v in q if v['sample']==name];ax.plot([v['bitrate_kbps'] for v in vals],[v['psnr_y'] for v in vals],'k.-',label='Uniform');ax.legend(fontsize=6);ax.set_title(name);ax.set_xlabel('Actual kbps');ax.set_ylabel('PSNR-Y dB')
    fig('mixed_precision_rd_curve');shutil.copyfile(out/'mixed_precision_rd_curve.png',out/'bitrate_vs_quality.png')
    policies=json.loads((ROOT/'configs/frozen_mixed_policies.json').read_text())['policies'];heat=[];names=[]
    for name in sorted({v['sample'] for v in mixed if v['split']=='validation'}):
        for policy in policies:
            v=next(v for v in mixed if v['sample']==name and v['policy']==policy);heat.append(v['channel_bits']);names.append(name+' '+policy['method'])
    plt.figure(figsize=(9,12));plt.imshow(heat,aspect='auto',vmin=2,vmax=8);plt.colorbar(label='Bits');plt.xlabel('Channel');plt.ylabel('Validation video / frozen policy');fig('bit_allocation_heatmap')
    gaps=[]
    for name in ('sample_0000','sample_1004'):
        oracle=[v for v in mixed if v['sample']==name and v['policy']['method']=='M4_approximate_oracle']
        for budget in sorted({v['policy']['budget'] for v in oracle}):
            o=max([v for v in oracle if v['policy']['budget']==budget and v['total_bytes']<=budget],key=lambda v:v['psnr_y'])
            feasible=[v for v in mixed if v['sample']==name and v['policy']['method'] in ('M1','M2','M2_three','M3') and v['total_bytes']<=budget]
            if feasible:
                best=max(feasible,key=lambda v:v['psnr_y']);gaps.append(dict(sample=name,budget_bytes=budget,approx_oracle_bytes=o['total_bytes'],approx_oracle_psnr=o['psnr_y'],best_deployable_bytes=best['total_bytes'],best_deployable_psnr=best['psnr_y'],oracle_minus_deployable_db=o['psnr_y']-best['psnr_y'],note='restricted greedy search can be worse; no global upper bound guarantee'))
    table(ROOT/'results/E1_2/oracle_gap.csv',gaps)
    reuse=json.loads((ROOT/'results/E1_3/skip_probe_results.json').read_text());correlations=[]
    for name in sorted({v['sample'] for v in reuse}):
        for size in sorted({v['block_size'] for v in reuse if v['sample']==name}):
            vals=[v for v in reuse if v['sample']==name and v['block_size']==size and v['configuration']['operation']=='int4']
            x=[v['mean_selected_residual_energy'] for v in vals];y=[10**(-v['psnr_rgb']/10)*255**2 for v in vals]
            correlations.append(dict(sample=name,block_size=size,n=len(vals),operation='quartile_INT4',pearson=corr(x,y),spearman=corr(ranks(x),ranks(y)),interpretation='exploratory within-video four-group association; not causal validation'))
    table(ROOT/'results/E1_3/energy_distortion_correlation.csv',correlations)
    figobj,axes=plt.subplots(3,3,figsize=(13,8))
    for i,name in enumerate(('sample_0000','sample_1004','sample_1032')):
        examples=[next(v for v in reuse if v['sample']==name and v['block_size']==4 and v['configuration']==dict(operation='reuse',ratio=.1,strategy=strategy,mode='closed_loop')) for strategy in ('low','random','high')]
        maximum=max(float(np.max(v['spatial_pixel_mse_32'])) for v in examples)
        for j,(strategy,v) in enumerate(zip(('low','random','high'),examples)):
            ax=axes[i,j];im=ax.imshow(v['spatial_pixel_mse_32'],vmin=0,vmax=maximum,cmap='magma');ax.set_title(name+' '+strategy,fontsize=8);figobj.colorbar(im,ax=ax)
    fig('local_pixel_distortion_heatmaps')
    for mode in ('single','closed_loop'):
        vals=[v for v in reuse if v['block_size']==4 and v['configuration']==dict(operation='reuse',ratio=.1,strategy='low',mode=mode)]
        plt.plot(np.mean([v['per_frame_psnr_y'] for v in vals],axis=0),'o-',label=mode)
    plt.xlabel('Video frame');plt.ylabel('Mean frame PSNR-Y dB');plt.legend();fig('temporal_error_propagation')
    latency=json.loads((ROOT/'results/latency_breakdown.json').read_text())
    names=[v['sample']+' '+Path(v['stream']).stem for v in latency]
    bottom=np.zeros(len(latency));plt.figure(figsize=(13,6))
    for key in ('quantization_s','packing_s','entropy_s'):
        values=np.array([v['timings'][key]['mean']*1000 for v in latency]);plt.bar(range(len(latency)),values,bottom=bottom,label=key);bottom+=values
    other=np.array([v['timings']['total_s']['mean']*1000 for v in latency])-bottom
    plt.bar(range(len(latency)),np.maximum(other,0),bottom=bottom,label='preparation/header/hash/I-O')
    plt.xticks(range(len(latency)),names,rotation=45,ha='right');plt.ylabel('CPU milliseconds');plt.legend();fig('latency_breakdown')
    for name in ('sample_0000','sample_1004'):
        vals=[v for v in latency if v['sample']==name and Path(v['stream']).stem.startswith('Q')]
        plt.plot([int(Path(v['stream']).stem[1:]) for v in vals],[v['timings']['total_s']['mean']*1000 for v in vals],'o-',label=name)
    plt.xlabel('Uniform bit width');plt.ylabel('Full CPU encode ms (warmup1 + repeats5)');plt.legend();fig('bitwidth_vs_latency')

from pathlib import Path
if __name__=='__main__':main()
