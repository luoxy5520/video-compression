"""Optional inference-only equivalent of CompressAI1.2.8 Gaussian index lookup.
For sorted table b, count(b[:-1] < scale) = min(lower_bound(b,scale),len(b)-1).
No change to probabilities, integer CDFs, neural weights or the stream format.
"""
import types,torch

@torch.no_grad()
def bucketize_indexes(self,scales):
    scales=self.lower_bound_scale(scales)
    if self.scale_table.ndim!=1 or not self.scale_table.numel():raise ValueError('populated scale table required')
    return torch.bucketize(scales.contiguous(),self.scale_table,right=False).clamp_max(self.scale_table.numel()-1).int()

def install_bucketize(model):
    gc=model.base.gc if hasattr(model,'base') else model.gc
    gc.build_indexes=types.MethodType(bucketize_indexes,gc)

