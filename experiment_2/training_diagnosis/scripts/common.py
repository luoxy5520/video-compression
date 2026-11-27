"""Diagnosis helpers; original glue around frozen historical E2 and CompressAI APIs."""
import sys,json,csv,hashlib,os,random,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; E2=ROOT.parent
sys.path.insert(0,str(E2))
from codec import Intra,load_model,write_stream,read_stream,sha
# Prevent oversubscription in independent decoder processes.
torch.set_num_threads(4)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def csvsave(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def atomic_torch(p,x):
    tmp=Path(str(p)+'.tmp');torch.save(x,tmp);os.replace(tmp,p)
def data():
    records=json.loads((E2/'dataset_manifest.json').read_text())['records'];train=[r for r in records if r['role']=='train_candidate' and r['exists']];return train

def tensor(record):
    z=torch.load(record['path'],weights_only=True,map_location='cpu');z=z[0] if z.ndim==5 else z;return z.permute(1,0,2,3).contiguous()
def forward(m,x,mode='noise'):
    y=m.ga(x);z=m.ha(y)
    median=m.eb._get_medians().reshape(1,m.channels,1,1)
    if mode=='noise':zh,_=m.eb(z,training=True)
    elif mode=='none':zh=z;yh=y
    else:
        zh=(z-median).round()+median;yh=y.round()
        if mode=='ste':zh=z+(zh-z).detach();yh=y+(yh-y).detach()
    scales=m.hs(zh)[:,:,:y.shape[-2],:y.shape[-1]]+.11
    pz=m.eb._likelihood(zh.permute(1,0,2,3).reshape(m.channels,1,-1))[0]
    if mode=='noise':yh,py=m.gc(y,scales,training=True)
    else:py=m.gc._likelihood(yh,scales)
    rec=m.gs(yh)[:,:,:x.shape[-2],:x.shape[-1]]
    main=-py.clamp_min(1e-9).log2().sum();hyper=-pz.clamp_min(1e-9).log2().sum()
    return dict(rec=rec,y=y,z=z,yh=yh,zh=zh,py=py,pz=pz,main=main,hyper=hyper,rate=(main+hyper)/x.numel(),dist=(rec-x).square().mean())
def optimizers(m,lr=.001):
    main=[p for n,p in m.named_parameters() if not n.endswith('.quantiles')];aux=[p for n,p in m.named_parameters() if n.endswith('.quantiles')]
    return torch.optim.Adam(main,lr=lr),torch.optim.Adam(aux,lr=.001),main
