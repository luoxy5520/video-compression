"""A0.1: frozen final latent -> binary P-macroblock mode; requires real oracle labels."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class ModeHead(nn.Module):
    def __init__(self,c=32):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(16,c,3,padding=1),nn.SiLU(),nn.Conv3d(c,2,1))
    def forward(self,x):return F.interpolate(self.net(x),size=(17,30,52),mode='trilinear',align_corners=False)[:,:,1:]
def metrics(pred,y):
    cm=torch.bincount((y.flatten()*2+pred.flatten()).cpu(),minlength=4).reshape(2,2).float()
    tp=cm.diag(); recall=tp/cm.sum(1).clamp_min(1); precision=tp/cm.sum(0).clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),skip_recall=float(recall[0]),p16x16_recall=float(recall[1]),support=cm.sum(1).int().tolist(),confusion_matrix=cm.int().tolist())
def load(cfg,ids):
    data=[]
    for sid in ids:
        folder=Path(cfg['labels_root'])/sid
        meta=json.loads((folder/'metadata.json').read_text())
        required=dict(sample_id=sid,source='solver_final',qp=28,pixel_format='yuv420p',profile='baseline',entropy='cavlc',reference_frames=1)
        for k,v in required.items():
            if meta.get(k)!=v:raise ValueError(f'{sid}: metadata {k} must be {v}')
        if not meta.get('backend_version'):raise ValueError('backend_version required')
        with np.load(folder/'state.npz',allow_pickle=False) as z:y=z['mode'].copy()
        if y.shape!=(16,30,52) or not np.isin(y,[0,1]).all():raise ValueError(f'{sid}: invalid mode')
        x=torch.load(Path(cfg['data_root'])/sid/'final_latent.pt',map_location='cpu',weights_only=True).float()
        if tuple(x.shape)!=(16,5,60,104) or not torch.isfinite(x).all():raise ValueError('invalid latent')
        data.append((sid,x.unsqueeze(0),torch.from_numpy(y.astype('int64')).unsqueeze(0)))
    return data
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
assert set(cfg['train_ids']).isdisjoint(cfg['test_ids'])
random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
# Keep prompt groups separate: car prompts train, ball prompts test.
train=load(cfg,cfg['train_ids']);test=load(cfg,cfg['test_ids'])
if a.check_only:print('Oracle label contract and latent validation PASS');raise SystemExit
out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu';model=ModeHead(cfg['channels']).to(device)
if a.evaluate:model.load_state_dict(torch.load(out/'mode_head.pt',map_location=device,weights_only=True))
else:
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    for epoch in range(cfg['epochs']):
        model.train();losses=[]
        for _,x,y in train:
            opt.zero_grad();loss=F.cross_entropy(model(x.to(device)),y.to(device));loss.backward();opt.step();losses.append(loss.item())
        print(epoch+1,sum(losses)/len(losses),flush=True)
    torch.save(model.cpu().state_dict(),out/'mode_head.pt');model.to(device)
model.eval();preds=[];ys=[];confidence=[]
with torch.no_grad():
    for sid,x,y in test:
        prob=model(x.to(device)).softmax(1).cpu();conf,pred=prob.max(1)
        np.savez_compressed(out/f'{sid}_pred.npz',mode=pred[0].byte().numpy(),confidence_mode=conf[0].numpy())
        preds.append(pred);ys.append(y);confidence.append(conf)
y=torch.cat(ys);result=metrics(torch.cat(preds),y)
counts=torch.bincount(torch.cat([r[2].flatten() for r in train]),minlength=2);majority=int(counts.argmax())
result.update(mean_confidence=float(torch.cat(confidence).mean()),majority_baseline=metrics(torch.full_like(y,majority),y),evaluation_scope='4-sample pipeline pilot; not evidence of generalization')
(out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(result,indent=2))
