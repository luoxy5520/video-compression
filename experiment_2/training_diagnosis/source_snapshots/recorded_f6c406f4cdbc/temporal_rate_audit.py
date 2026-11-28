"""Separate continuous likelihood, represented scale-table likelihood and real payload."""
from common import *
from temporal_entropy_model import load_temporal
out=ROOT/'results/E2_2';measured=json.loads((out/'temporal_results.json').read_text());records=json.loads((E2/'dataset_manifest.json').read_text())['records'];rows=[]
with torch.no_grad():
    for mode in ('TP0','TP1','TP2'):
        m,c=load_temporal(ROOT/'checkpoints/temporal'/f'{mode}.pt')
        for row in [r for r in measured if r['mode']==mode]:
            record=next(r for r in records if r['path']==row['video']);fixed=m.fixed_sequence(tensor(record));prev=None
            for t,(q,hs,zs,zshape) in enumerate(fixed):
                mean,scale=m.params_for(hs,prev);indexes=m.base.gc.build_indexes(scale);quantized_scale=m.base.gc.scale_table[indexes.long()];p=m.base.gc._likelihood(q,scale,mean);pq=m.base.gc._likelihood(q,quantized_scale,mean);zh=m.base.eb.decompress([zs],zshape);pz=m.base.eb._likelihood(zh.permute(1,0,2,3).reshape(m.base.channels,1,-1))[0];actual=row['slices'][t]
                rows.append(dict(mode=mode,video=row['video'],slice=t,estimated_main_bits=float(-p.clamp_min(1e-9).log2().sum()),estimated_main_scale_table_bits=float(-pq.clamp_min(1e-9).log2().sum()),estimated_hyper_bits=float(-pz.clamp_min(1e-9).log2().sum()),actual_main_bits=actual['main']*8,actual_hyper_bits=actual['hyper']*8,metadata_bits_video=row['metadata_bytes']*8,mean_min=float(mean.min()),mean_max=float(mean.max()),scale_min=float(scale.min()),scale_max=float(scale.max()),scale_above_table_fraction=float((scale>m.base.gc.scale_table[-1]).float().mean()),zero_main_likelihoods=int((p==0).sum()),nonfinite_likelihoods=int((~torch.isfinite(p)).sum()+(~torch.isfinite(pz)).sum())))
                prev=m.previous(q)
assert all(r['nonfinite_likelihoods']==0 for r in rows)
csvsave(out/'estimated_vs_actual_temporal_rate.csv',rows);print('audited',len(rows),'slice rows; max overflow fraction',max(r['scale_above_table_fraction'] for r in rows))
