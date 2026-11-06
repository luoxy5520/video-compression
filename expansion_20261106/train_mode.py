"""Real Oracle A0.1/A2-final curriculum; grouped fixed split, validation checkpoint selection."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class Head(nn.Module):
    def __init__(self,c=32):
        super().__init__();self.conv=nn.Conv3d(16,c,3,padding=1);self.t=nn.Sequential(nn.Linear(1,c),nn.SiLU(),nn.Linear(c,c));self.out=nn.Conv3d(c,2,1)
    def forward(self,x,s):return F.interpolate(self.out(F.silu(self.conv(x)+self.t(s.reshape(-1,1))[:,:,None,None,None])),size=(17,30,52),mode='trilinear',align_corners=False)[:,:,1:]
def metric(pred,y):
    cm=torch.bincount((2*y.flatten()+pred.flatten()).cpu(),minlength=4).reshape(2,2).float();tp=cm.diag();rec=tp/cm.sum(1).clamp_min(1);pre=tp/cm.sum(0).clamp_min(1);f=2*rec*pre/(rec+pre).clamp_min(1e-12)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f.mean()),skip_recall=float(rec[0]),p16_recall=float(rec[1]),confusion_matrix=cm.int().tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    dataset=Path(cfg['dataset']);rows=[json.loads(x) for x in (dataset/'manifests/all.jsonl').read_text().splitlines()];splits=cfg['splits'];ids=[s for v in splits.values() for s in v];assert len(ids)==len(set(ids))
    prompt_sets=[{r['prompt'] for r in rows if r['sample_id'] in v} for v in splits.values()]
    for i,u in enumerate(prompt_sets):
        for v in prompt_sets[i+1:]:assert not u&v,'prompt leakage'
    cache={};labels={}
    for r in rows:
        if r['sample_id'] not in ids:continue
        x=torch.load(r['latent_path'],map_location='cpu',weights_only=True).float().unsqueeze(0);assert torch.isfinite(x).all()
        key=r['target_label_path']
        if key not in labels:
            with np.load(dataset/key) as z:labels[key]=torch.from_numpy(z['mode_id'].astype('int64')).unsqueeze(0)
        cache[(r['sample_id'],int(r['input_step']))]=(x,labels[key])
    steps=[20,15,10,5]
    for sid in ids:
        for s in steps:assert (sid,s) in cache
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);device='cuda';m=Head().to(device);opt=torch.optim.Adam(m.parameters(),lr=cfg.get('lr',.001));history=[];report={}
    counts=torch.bincount(torch.cat([cache[sid,20][1].flatten() for sid in splits['train']]),minlength=2);majority=int(counts.argmax())
    def evaluate(ids,ss,save=False):
        m.eval();result={}
        with torch.no_grad():
            for s in ss:
                ps=[];ys=[];cs=[]
                for sid in ids:
                    x,y=cache[sid,s];probs=m(x.to(device),torch.tensor([s/20],device=device)).softmax(1).cpu();c,pred=probs.max(1);ps.append(pred);ys.append(y);cs.append(c)
                    if save:np.savez_compressed(out/f'{sid}_step_{s:03d}_mode.npz',mode_id=pred[0].byte().numpy(),confidence_mode=c[0].numpy())
                if ps:
                    y=torch.cat(ys);result[str(s)]=metric(torch.cat(ps),y);result[str(s)].update(mean_confidence=float(torch.cat(cs).mean()),majority_baseline=metric(torch.full_like(y,majority),y))
        return result
    for stage in range(4):
        active=steps[:stage+1];best=-1.;checkpoint=out/f'stage_{stage+1}.pt'
        for epoch in range(cfg.get('epochs_per_stage',10)):
            m.train();keys=[(sid,s) for sid in splits['train'] for s in active];random.shuffle(keys);losses=[]
            for sid,s in keys:
                x,y=cache[sid,s];opt.zero_grad();logits=m(x.to(device),torch.tensor([s/20],device=device));loss=F.cross_entropy(logits,y.to(device));loss.backward();opt.step();losses.append(loss.item())
            val=evaluate(splits.get('val',[]),active);score=float(np.mean([v['macro_f1'] for v in val.values()])) if val else -float(np.mean(losses))
            history.append(dict(stage=stage+1,epoch=epoch+1,train_loss=float(np.mean(losses)),val=val));print(json.dumps(history[-1]),flush=True)
            if score>best or (not checkpoint.exists()):best=score;torch.save(m.state_dict(),checkpoint)
        m.load_state_dict(torch.load(checkpoint,weights_only=True));report[f'stage_{stage+1}']=dict(steps=active,validation=evaluate(splits.get('val',[]),active),train=evaluate(splits['train'],active))
        # A0 checkpoint is stage1. A2 stages add earlier clean-latent inputs.
    report['test_by_step']=evaluate(splits['test'],steps,True)
    # Evaluate A0 checkpoint once on held-out final-step features.
    final_state={k:v.detach().clone() for k,v in m.state_dict().items()};m.load_state_dict(torch.load(out/'stage_1.pt',weights_only=True));report['a0_test']=evaluate(splits['test'],[20]);m.load_state_dict(final_state)
    report.update(parameters=sum(p.numel() for p in m.parameters()),target='final P-frame modes under Oracle-I',sample_count=len(ids),label_source='real JM oracle',limitation='pilot only; mode predictions do not include MV/residual and cannot alone replay H264',config=cfg)
    (out/'history.json').write_text(json.dumps(history,indent=2));(out/'metrics.json').write_text(json.dumps(report,indent=2));print('TRAIN_COMPLETE',out,flush=True)
if __name__=='__main__':main()
