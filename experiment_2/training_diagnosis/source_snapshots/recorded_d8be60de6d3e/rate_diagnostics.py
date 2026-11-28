"""Continuous/table likelihood diagnostics beside actual temporal rANS bytes."""
from common import torch

@torch.no_grad()
def temporal_rates(model,fixed,actual_slices):
    rows=[];previous=None
    for t,((q,hs,zstream,zshape),actual) in enumerate(zip(fixed,actual_slices,strict=True)):
        mean,scale=model.params_for(hs,previous)
        indexes=model.base.gc.build_indexes(scale)
        table_scale=model.base.gc.scale_table[indexes.long()]
        p=model.base.gc._likelihood(q,scale,mean)
        table_p=model.base.gc._likelihood(q,table_scale,mean)
        zh=model.base.eb.decompress([zstream],zshape)
        pz=model.base.eb._likelihood(zh.permute(1,0,2,3).reshape(model.base.channels,1,-1))[0]
        for mass in (p,table_p,pz):
            assert torch.isfinite(mass).all() and (mass>=0).all() and (mass<=1).all()
        assert torch.isfinite(mean).all() and torch.isfinite(scale).all()
        rows.append(dict(time_slice=t,
            estimated_main_bits=float(-p.clamp_min(1e-9).log2().sum()),
            estimated_main_scale_table_bits=float(-table_p.clamp_min(1e-9).log2().sum()),
            estimated_hyper_bits=float(-pz.clamp_min(1e-9).log2().sum()),
            actual_main_bits=actual['main']*8,actual_hyper_bits=actual['hyper']*8,
            mean_min=float(mean.min()),mean_max=float(mean.max()),
            scale_min=float(scale.min()),scale_max=float(scale.max()),
            scale_above_table_fraction=float((scale>model.base.gc.scale_table[-1]).float().mean()),
            scale_below_table_fraction=float((scale<model.base.gc.scale_table[0]).float().mean()),
            zero_main_likelihoods=int((p==0).sum()),
            likelihood_floor_main_count=int((p<1e-9).sum()),
            likelihood_floor_hyper_count=int((pz<1e-9).sum()),nonfinite_likelihoods=0))
        previous=model.previous(q)
    return rows
