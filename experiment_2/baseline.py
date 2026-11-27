"""Complete exact E1-style uniform development curve, CPU encoder only."""
import json,importlib.util,torch
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e1_lvc',R.parent/'experiment_1/codec/lvc.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
records=json.loads((R/'dataset_manifest.json').read_text())['records'];train=[r for r in records if r['role']=='train_candidate' and r['exists']][:16];groups={r['group'] for r in train};dev=[r for r in records if r['role']=='historical_development' and r['exists'] and r['group'] not in groups][:2]
rows=[]
for idx,r in enumerate(dev):
    x=torch.load(r['path'],weights_only=True).numpy();x=x[None] if x.ndim==4 else x
    for bits in range(2,9):
        p=R/'results'/f'I0_int{bits}_{idx}.lvc';stats=c.encode(x,p,dict(bits=bits,granularity='channel',percentile=100),dict(algorithm='zstd'),dict(width=832,height=480,frames=17,fps=16));xh=c.decode(p)
        rows.append(dict(sample=r['path'],bits=bits,**stats,latent_mse=float(((xh-x)**2).mean())))
(R/'results/uniform.json').write_text(json.dumps(rows,indent=2))
