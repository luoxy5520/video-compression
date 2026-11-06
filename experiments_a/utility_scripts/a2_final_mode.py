"""Time-conditioned shared head: intermediate clean latent -> final oracle modes."""
import argparse,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FinalModeHead(nn.Module):
    def __init__(self,channels=32):
        super().__init__()
        self.features=nn.Conv3d(16,channels,3,padding=1)
        self.time=nn.Sequential(nn.Linear(1,channels),nn.SiLU(),nn.Linear(channels,channels))
        self.output=nn.Conv3d(channels,2,1)
    def forward(self,x,progress):
        h=self.features(x)+self.time(progress.reshape(-1,1).to(x))[:,:,None,None,None]
        return F.interpolate(self.output(F.silu(h)),size=(17,30,52),mode='trilinear',align_corners=False)[:,:,1:]

def scores(pred,y):
    cm=torch.bincount((y.flatten()*2+pred.flatten()).cpu(),minlength=4).reshape(2,2).float()
    tp=cm.diag();recall=tp/cm.sum(1).clamp_min(1);precision=tp/cm.sum(0).clamp_min(1)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float((2*precision*recall/(precision+recall).clamp_min(1e-12)).mean()),skip_recall=float(recall[0]),p16x16_recall=float(recall[1]),confusion_matrix=cm.int().tolist())

def read_records(cfg):
    root=Path(cfg['data_root']);obj=json.loads((root/'a2_final_pairs.json').read_text());rows=obj['records']
    train=set(cfg['train_ids']);test=set(cfg['test_ids'])
    if train & test or not train or not test:raise ValueError('Empty/overlapping split')
    prompts=lambda ids:{r['prompt'] for r in rows if r['sample_id'] in ids}
    if prompts(train)&prompts(test):raise ValueError('Prompt leakage')
    for sid in train|test:
        if sorted(r['step'] for r in rows if r['sample_id']==sid)!=cfg['steps']:raise ValueError(f'Missing steps: {sid}')
    return rows

def load(cfg,rows,ids,require_labels=True):
    root=Path(cfg['data_root']);labels=Path(cfg['labels_root']);result=[];cache={}
    for r in rows:
        sid=r['sample_id']
        if sid not in ids:continue
        x=torch.load(root/r['latent'],map_location='cpu',weights_only=True).float()
        if x.shape!=(16,5,60,104) or not torch.isfinite(x).all():raise ValueError('Invalid latent')
        y=None
        if require_labels:
            if sid not in cache:
                meta=json.loads((labels/r['target_metadata']).read_text())
                for k,v in dict(sample_id=sid,source='solver_final',qp=28,pixel_format='yuv420p',profile='baseline',entropy='cavlc',reference_frames=1,raw_sha256=r['target_raw_sha256']).items():
                    if meta.get(k)!=v:raise ValueError(f'{sid}: metadata mismatch {k}')
                if not meta.get('backend_version'):raise ValueError('Missing backend version')
                with np.load(labels/r['target_label'],allow_pickle=False) as z:mode=z['mode'].copy()
                if mode.shape!=(16,30,52) or not np.isin(mode,[0,1]).all():raise ValueError('Invalid mode label')
                cache[sid]=torch.from_numpy(mode.astype('int64')).unsqueeze(0)
            y=cache[sid]
        result.append((r,x.unsqueeze(0),y))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check-features',action='store_true');p.add_argument('--check-labels',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    rows=read_records(cfg)
    train=load(cfg,rows,cfg['train_ids'],not a.check_features);test=load(cfg,rows,cfg['test_ids'],not a.check_features)
    if a.check_features or a.check_labels:print(f'PASS: {len(train)} train / {len(test)} test records; grouped splits; label verification={not a.check_features}');return
    device='cuda' if torch.cuda.is_available() else 'cpu';model=FinalModeHead(cfg['channels']).to(device);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if a.evaluate:model.load_state_dict(torch.load(out/'head.pt',map_location=device,weights_only=True))
    else:
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        for epoch in range(cfg['epochs']):
            model.train();order=list(train);random.shuffle(order);losses=[]
            for r,x,y in order:
                opt.zero_grad();logit=model(x.to(device),torch.tensor([r['step']/r['total_steps']],device=device));loss=F.cross_entropy(logit,y.to(device));loss.backward();opt.step();losses.append(loss.item())
            print(epoch+1,float(np.mean(losses)),flush=True)
        torch.save(model.cpu().state_dict(),out/'head.pt');model.to(device)
    counts=torch.bincount(torch.cat([y.flatten() for r,x,y in train if r['step']==cfg['steps'][0]]),minlength=2);majority=int(counts.argmax());report={};model.eval()
    with torch.no_grad():
        for step in cfg['steps']:
            preds=[];ys=[];confs=[]
            for r,x,y in test:
                if r['step']!=step:continue
                prob=model(x.to(device),torch.tensor([step/r['total_steps']],device=device)).softmax(1).cpu();conf,pred=prob.max(1)
                np.savez_compressed(out/f'{r["sample_id"]}_step_{step:03d}_final_pred.npz',mode=pred[0].byte().numpy(),confidence_mode=conf[0].numpy())
                preds.append(pred);ys.append(y);confs.append(conf)
            y=torch.cat(ys);report[str(step)]=scores(torch.cat(preds),y)
            report[str(step)].update(mean_confidence=float(torch.cat(confs).mean()),majority_baseline=scores(torch.full_like(y,majority),y))
    (out/'metrics_by_step.json').write_text(json.dumps(report,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
