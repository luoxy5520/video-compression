import json, time, sys
sys.path.insert(0,str(__import__("pathlib").Path.home()/"dit-paired-experiment/a_state_20261107"))
from motion_comp import luma as mc_luma
from pathlib import Path
import numpy as np
import torch
from baseline import KITS, sha

DATA = Path('/mnt/data_hdd/luoxiangyu/dit_pairs')
SETS = ('exp_expand32_20261106', 'exp_a_new48_20261107')
H, W = 30, 52

def match(cur, ref, radius=4):
    h,w = cur.shape[:2]
    offsets = sorted(((x,y) for y in range(-radius,radius+1)
                      for x in range(-radius,radius+1)), key=lambda d:d[0]**2+d[1]**2)
    cost = np.full((h,w,len(offsets)), np.inf, np.float32)
    for i,(x,y) in enumerate(offsets):
        a,b=max(0,-y),min(h,h-y)
        c,d=max(0,-x),min(w,w-x)
        cost[a:b,c:d,i]=np.square(cur[a:b,c:d]-ref[a+y:b+y,c+x:d+x]).mean(axis=-1)
    return np.asarray(offsets)[np.argsort(cost,axis=-1,kind='stable')[...,:8]]

def normalization(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8)

def tensor(path):
    return torch.load(path,map_location='cpu',weights_only=True)

def unique(candidates):
    # Maintain order and compare only distinct qpel candidate centers.
    flat=candidates.reshape(-1,candidates.shape[-2],2)
    out=np.full((len(flat),8,2),100000,np.int32)
    n=np.zeros(len(flat),np.int32)
    for c in flat.transpose(1,0,2):
        use=(n<8)&~(out==c[:,None]).all(axis=-1).any(axis=-1)
        out[use,n[use]]=c[use]
        n+=use
    return out

def luma(path):
    return np.fromfile(path,np.uint8).reshape(17,-1)[:,:480*832].reshape(17,480,832)

def analyze(sample, split):
    start=time.perf_counter()
    dataset=int(int(sample.split('_')[1])>=1000)
    root=DATA/SETS[dataset]/sample
    old=KITS[dataset]/'output/oracle_exports'/sample/'step_020'
    meta=json.loads((old/'metadata.json').read_text())
    assert meta['pairing']['sample_id']==sample and meta['h264']['reference_frames']==1
    raw=old/'input_yuv420p.yuv'
    assert sha(raw)==sha(root/'final_raw.yuv')==meta['input']['sha256']
    latent=tensor(root/'final_latent.pt')
    assert tuple(latent.shape)==(16,5,60,104)
    assert torch.equal(latent,tensor(root/'step_020/clean_latent.pt'))
    hidden=tensor(root/'step_020/hidden_cond_last_block.pt')
    assert tuple(hidden.shape)==(1,7800,1536)
    z=latent.numpy().reshape(16,5,H,2,W,2).transpose(1,2,4,3,5,0).reshape(5,H,W,64)
    h=hidden.numpy().reshape(5,H,W,1536)
    with np.load(old/'oracle_labels.npz') as labels:
        mask=(labels['is_p'][1:]==1) & (labels['is_skip'][1:]==0) & (labels['mv_is_uniform'][1:]==1) & (labels['ref_idx_mb'][1:]==0)
        mv=labels['mv_qpel_mb'][1:].astype(np.int32)
    assert mask.shape==(16,H,W)
    left=np.pad(mv[:,:,:-1],((0,0),(0,0),(1,0),(0,0)))
    top=np.pad(mv[:,:-1],((0,0),(1,0),(0,0),(0,0)))
    right=np.pad(mv[:,:-1,1:],((0,0),(1,0),(0,1),(0,0)))
    prev=np.concatenate([np.zeros_like(mv[:1]),mv[:-1]])
    native=np.stack([np.median([left,top,right],axis=0).astype(np.int32),
                     np.zeros_like(mv),left,top,right,prev],axis=-2)
    mvp_file=Path('trace')/sample/'mvp.tsv'
    assert mvp_file.is_file(),mvp_file
    trace=np.loadtxt(mvp_file,dtype=np.int32)
    assert trace.shape==(16*H*W,5)
    f,x,y=trace[:,:3].T
    idx=(f-1)*H*W+(y//16)*W+x//16
    assert np.array_equal(np.sort(idx),np.arange(16*H*W))
    assert (x%16==0).all() and (y%16==0).all()
    mvp=np.empty_like(mv).reshape(-1,2)
    mvp[idx]=trace[:,3:]
    mvp=mvp.reshape(16,H,W,2)
    native=np.concatenate([mvp[...,None,:],native[...,1:,:]],axis=-2)
    arms={'E0':unique(native[mask])}
    timing={}
    for name, feature in [('E2',z),('E3',h)]:
        t=time.perf_counter()
        feature=normalization(feature)
        offsets=np.stack([match(feature[j],feature[j-1]) for j in range(1,5)])
        # 16 px/grid cell / 4 frame interval × 4 qpel/px.
        c=(16*offsets).repeat(4,axis=0)
        arms[name]=unique(c[mask])
        arms[name+'+E0']=unique(np.concatenate([native[...,:1,:],c[...,:1,:],native[...,1:,:],c[...,1:,:]],axis=-2)[mask])
        timing[name]=time.perf_counter()-t
    t=time.perf_counter()
    source=luma(raw)
    recon=luma(old/'reference_recon.yuv')
    # Four by four luma means per macroblock, adjacent reconstructed frame.
    small=lambda a:a.reshape(17,H,4,4,W,4,4).mean(axis=(3,6)).transpose(0,1,3,2,4).reshape(17,H,W,16).astype(np.float32)
    a,b=small(source),small(recon)
    c=np.stack([match(a[j],b[j-1]) for j in range(1,17)])*64
    arms['E5+E0']=unique(np.concatenate([native[...,:1,:],c[...,:1,:],native[...,1:,:],c[...,1:,:]],axis=-2)[mask])
    timing['E5']=time.perf_counter()-t
    head=Path.home()/'a_handoff80_20261107/predictions'/split/sample/'step_020/prediction.npz'
    mode_stats={}
    if head.exists():
        with np.load(head) as p:
            arms['E1-cached']=unique(p['mv_qpel'][mask][:,None])
            predicts=p['mode_id']==1
            assert predicts.shape==mask.shape
            mode_stats=dict(accuracy=float((predicts==mask).mean()),
                            p16_recall=float(predicts[mask].mean()),
                            skip_recall=float((~predicts[~mask]).mean()))
    def sad(p):
        values=[]
        expanded=np.zeros((16,H,W,2),np.int32)
        expanded[mask]=np.where(p==100000,0,p)
        for j in range(16):
            pred=mc_luma(recon[j],expanded[j])
            cost=np.abs(source[j+1].astype(np.int16)-pred)
            values.append(cost.reshape(H,16,W,16).sum(axis=(1,3)))
        return np.asarray(values)[mask]
    anchor=sad(mv[mask])
    stats={}
    for name,p in arms.items():
        cost=np.stack([sad(p[:,i]) for i in range(2)],axis=1)
        cost[p[:,:2,0]==100000]=10**8
        regret=cost.min(axis=1)-anchor
        difference=p-mv[mask][:,None]
        d=np.abs(difference).max(axis=2)
        stats[name]={str(k):{'coverage':[np.mean(d[:,:k].min(axis=1)<=r*4).item() for r in (0,2,4,8)],
                            'unique':(p[:,:k,0]!=100000).sum(axis=1).mean().item(),
                            'epe_median_p90':np.percentile(np.linalg.norm(difference[:,:k],axis=-1).min(axis=1)/4,[50,90]).tolist()}
                     for k in (1,2,4,8)}
        stats[name]['sad_k2']={'mean':float(cost.min(axis=1).mean()),'regret_mean':float(regret.mean())}

    return dict(sample=sample,prompt=meta['pairing']['prompt'],eligible=int(mask.sum()),p_blocks=mask.size,mode_stats=mode_stats,stats=stats,timing=timing,wall=time.perf_counter()-start)

if __name__=='__main__':
    rng=np.random.default_rng(9)
    ref=rng.standard_normal((12,12,8)).astype(np.float32)
    assert (match(ref,ref)[...,0,:]==0).all()
    shifted=np.roll(ref,1,axis=1)
    assert (match(shifted,ref)[1:-1,2:-1,0,:]==[-1,0]).all()
    assert tuple(16*np.array([-1,0]))==(-16,0)
    split=sys.argv[1]
    for sample in sys.argv[2:]:
        print(json.dumps(analyze(sample,split)),flush=True)
