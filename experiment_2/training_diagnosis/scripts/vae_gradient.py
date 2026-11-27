from pixels import *
from diagnostic_codec import load
import argparse,gc
p=argparse.ArgumentParser();p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--name',default='D16_f1_seed20261127');p.add_argument('--step',type=int,default=10000);a=p.parse_args()
assert json.loads((ROOT/'results/E2_1d/gate.json').read_text())['pass']
out=ROOT/'results/E2_1e';path=out/f'vae_gradient_{a.precision}.json';m,c=load(ROOT/'checkpoints'/a.name/f'step{a.step}.pt');m=m.cuda().train();x=tensor(c['sample']).cuda();v=vae();v.model.requires_grad_(False)
def fingerprint():
    h=hashlib.sha256()
    for n,p in v.model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
result=dict(precision=a.precision,sample=c['sample']['path'],full_sequence=True,shape=list(x.shape),checkpoint=a.name,vae_file_sha256_before=sha(MODEL),vae_parameters_frozen=all(not p.requires_grad for p in v.model.parameters()),vae_not_in_optimizer=True)
before=fingerprint();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
try:
    with torch.no_grad():
        native=decode(v,x);direct=v.model.decode(x.permute(1,0,2,3)[None],v.scale)[0].clamp(-1,1)
        result['native_direct_max_abs_difference']=(native-direct).abs().max().item();assert torch.equal(native,direct)
        reference=native.detach();del direct,native
    f=forward(m,x,'ste');zh=f['rec'];zh.retain_grad();torch.cuda.synchronize();t=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
        reconstruction=v.model.decode(zh.permute(1,0,2,3)[None],v.scale)[0].float().clamp(-1,1)
    torch.cuda.synchronize();result['vae_forward_seconds']=time.perf_counter()-t;loss=(reconstruction-reference).square().mean();t=time.perf_counter();loss.backward();torch.cuda.synchronize();result['backward_seconds']=time.perf_counter()-t
    result.update(status='PASS',pixel_mse=loss.item(),zhat_gradient_norm=zh.grad.norm().item(),zhat_gradient_norm_per_slice=zh.grad.flatten(1).norm(dim=1).tolist(),ga_gradient_norm=sum(p.grad.square().sum().item() for p in m.ga.parameters() if p.grad is not None)**.5,gs_gradient_norm=sum(p.grad.square().sum().item() for p in m.gs.parameters() if p.grad is not None)**.5,vae_gradient_tensors=sum(p.grad is not None for p in v.model.parameters()))
    assert result['zhat_gradient_norm']>0 and result['ga_gradient_norm']>0 and result['gs_gradient_norm']>0 and result['vae_gradient_tensors']==0
    if a.precision=='bf16':
        mixed_output=reconstruction.detach().cpu();del reconstruction;gc.collect();torch.cuda.empty_cache()
        with torch.no_grad():native_reconstruction=decode(v,zh.detach());result['fp32_reconstruction_pixel_mse']=(native_reconstruction-reference).square().mean().item();difference=mixed_output-native_reconstruction.cpu();result['bf16_vs_fp32_output_mse']=difference.square().mean().item();result['bf16_vs_fp32_max_abs_difference']=difference.abs().max().item()
except torch.OutOfMemoryError as e:result.update(status='OOM',error=str(e))
except Exception as e:result.update(status='FAILED',error=repr(e))
result['peak_gpu_bytes']=torch.cuda.max_memory_allocated();result['elapsed_seconds']=time.perf_counter()-started;result['vae_tensor_sha256_before']=before;result['vae_tensor_sha256_after']=fingerprint();result['vae_file_sha256_after']=sha(MODEL);assert result['vae_tensor_sha256_after']==before
save(path,result);print(json.dumps(result),flush=True)
