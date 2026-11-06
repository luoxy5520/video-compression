"""Audit completed expansion without treating correlated macroblocks as independent videos."""
from pathlib import Path
import json,re,hashlib
import torch
BASE=Path('/home/luoxiangyu/dit-paired-experiment/expansion_20261106');DATA=Path('/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106');KIT=Path('/home/luoxiangyu/senior-oracle-expand32-20261106');torch.set_num_threads(4)
rows=[json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines()];assert len(rows)==128 and len({(r['sample_id'],r['step']) for r in rows})==128
for r in rows:
    assert Path(r['raw_yuv']).stat().st_size==10183680
    for k,shape in [('latent',(16,5,60,104)),('hidden',(1,7800,1536))]:
        t=torch.load(r[k],map_location='cpu',weights_only=True);assert tuple(t.shape)==shape and torch.isfinite(t).all(),(r['sample_id'],r['step'],k)
closures=[]
for p in sorted((KIT/'output/oracle_exports').glob('*/step_*/metadata.json')):
    if not re.fullmatch('step_[0-9]{3}',p.parent.name):continue
    c=json.loads((p.parent/'closure_result.json').read_text());assert all(c.values())
    m=re.search(r'Total ME time for sequence\s*:\s*([0-9.]+)',(p.parent/'replay.log').read_text());assert m and float(m[1])==0
    closures.append(dict(sample_id=p.parent.parent.name,step=p.parent.name,validation=c))
assert len(closures)==128
index=[json.loads(x) for x in (KIT/'output/training_dataset/manifests/all.jsonl').read_text().splitlines()];assert len(index)==128
for sid in {r['sample_id'] for r in index}:
    assert len({r['target_label_path'] for r in index if r['sample_id']==sid})==1
portable=[]
for r in index:
    r=dict(r,dataset_id='exp_expand32_20261106')
    for k in ['latent_path','hidden_path','final_latent_path']:r[k]=str(Path(r[k]).relative_to(DATA))
    portable.append(r)
(DATA/'training_index.json').write_text(json.dumps(dict(feature_root='directory containing this JSON',label_root=str(KIT/'output/training_dataset'),records=portable),indent=2))
report=dict(status='PASS',videos=32,prompts=8,seeds_per_prompt=4,feature_records=128,unique_final_targets=32,oracle_closures=closures,splits=json.loads((BASE/'splits32.json').read_text()),model_result=str(DATA.parent/'training_expand32_20261106/metrics.json'),limitations=['Pilot only: 8 prompts, val only 1 prompt.','Modes only; no trained MV/residual head or online exit controller.','Oracle final I-frame condition.','Use (dataset_id,sample_id,step) when combining batches.'])
(BASE/'audit32.json').write_text(json.dumps(report,indent=2));print('AUDIT_EXPAND32_PASS')
