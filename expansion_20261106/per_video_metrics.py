"""Compute video-level test scores from saved predictions, with no retraining."""
from pathlib import Path
import json,numpy as np,torch
from train_mode import metric
ROOT=Path('/home/luoxiangyu/dit-paired-experiment/expansion_20261106');cfg=json.loads((ROOT/'config_train32.json').read_text());dataset=Path(cfg['dataset']);out=Path(cfg['output']);rows=[json.loads(x) for x in (dataset/'manifests/all.jsonl').read_text().splitlines()];report=[]
for sid in cfg['splits']['test']:
    row=next(r for r in rows if r['sample_id']==sid)
    with np.load(dataset/row['target_label_path']) as z:y=torch.from_numpy(z['mode_id'].astype('int64'))
    for step in [5,10,15,20]:
        with np.load(out/f'{sid}_step_{step:03d}_mode.npz') as z:pred=torch.from_numpy(z['mode_id'].astype('int64'))
        report.append(dict(sample_id=sid,step=step,prompt=row['prompt'],seed=row['seed'],**metric(pred,y)))
(out/'per_video_metrics.json').write_text(json.dumps(report,indent=2));print('step20 F1 range',min(r['macro_f1'] for r in report if r['step']==20),max(r['macro_f1'] for r in report if r['step']==20))
