"""Binary Skip/P16 coverage at a fixed 20% candidate reduction."""
import json,sys
from pathlib import Path
import numpy as np
from baseline import KITS
from probe import luma

train=['sample_0000','sample_0004','sample_0008','sample_0012',
       'sample_0016','sample_1000','sample_1004','sample_1008']
val=json.loads(Path('frozen.json').read_text())['validation']
minimum_class_recall=0.99

def inputs(sample, split):
    kit=KITS[int(int(sample.split('_')[1])>=1000)]
    folder=kit/'output/oracle_exports'/sample/'step_020'
    with np.load(folder/'oracle_labels.npz') as labels:
        assert labels['is_p'][1:].all()
        truth=labels['is_skip'][1:]==0
    source=luma(folder/'input_yuv420p.yuv')
    recon=luma(folder/'reference_recon.yuv')
    sad=np.abs(source[1:].astype(np.int16)-recon[:-1]).reshape(16,30,16,52,16)
    cost=sad.mean(axis=(2,4)).ravel()
    cache=Path.home()/'a_handoff80_20261107/predictions'/split/sample/'step_020/prediction.npz'
    with np.load(cache) as p:
        score=p['p16_probability'].ravel()
    prompt=json.loads((folder/'metadata.json').read_text())['pairing']['prompt']
    return sample,prompt,truth.ravel(),score,cost

def evaluate(truth, prediction, confidence):
    order=np.argsort(-confidence,kind='stable')
    keep_one=order[:int(0.4*len(truth))]
    miss=np.zeros_like(truth)
    miss[keep_one]=prediction[keep_one]!=truth[keep_one]
    return dict(candidates=2-len(keep_one)/len(order),coverage=float(1-miss.mean()),
                skip=float(1-miss[~truth].mean()),p16=float(1-miss[truth].mean()),
                chosen_solo=len(keep_one))

dev=[inputs(s,'train') for s in train]
thresholds=np.unique(np.percentile(np.concatenate([x[4] for x in dev]),np.arange(5,96,5)))
cost_threshold=max(thresholds,key=lambda t:np.mean([
    evaluate(y,c>=t,np.abs(c-t))['coverage'] for _,_,y,_,c in dev]))
print(json.dumps({'dev_m4_zero_sad_threshold':float(cost_threshold),
                  'minimum_class_recall':minimum_class_recall}),flush=True)
for split in sys.argv[1:]:
    rows=dev if split=='train' else [inputs(s,'val') for s in val]
    for sample,prompt,y,p,c in rows:
        print(json.dumps(dict(sample=sample,prompt=prompt,split=split,
             M1=evaluate(y,p>=0.5,np.abs(p-0.5)),
             M4=evaluate(y,c>=cost_threshold,np.abs(c-cost_threshold)))),flush=True)
