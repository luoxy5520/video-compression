"""Only inspect unsealed train/val metadata; never open sealed test tensors."""
import json,hashlib,platform,subprocess
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
root=Path('/mnt/data_hdd/luoxiangyu/dit_pairs')
historical=json.loads((Path.home()/'E0_Direct_Latent_Compression/experiment_1/results/input_manifest.json').read_text())
hist={str(Path(x['path']).resolve()) for x in historical}
records={}
for source in ('exp_expand32_20261106','exp_a_new48_20261107'):
    for line in (root/source/'manifest.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x.get('split') not in ('train','val','validation'): continue
        p=Path(x['final_latent']); key=str(p)
        if key in records: continue
        group=hashlib.sha256((x['prompt']+'\0'+str(x['seed'])).encode()).hexdigest()
        role='historical_development' if key in hist or x['split'] in ('val','validation') else 'train_candidate'
        records[key]=dict(path=key,source=source,prompt=x['prompt'],seed=x['seed'],original_split=x['split'],role=role,group=group,exists=p.exists(),reference=str(p.parent/'final_raw.yuv'))
# Metadata-only grouping audit. No claim that any candidate is an untouched holdout.
(R/'dataset_manifest.json').write_text(json.dumps(dict(policy='sealed test excluded, original val is historical development; no independent holdout certified',records=list(records.values())),indent=2))
info=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),compressai=__import__('compressai').__version__,wan_git=subprocess.check_output(['git','-C',str(Path.home()/'Wan2.1'),'rev-parse','HEAD'],text=True).strip())
(R/'results/environment.json').write_text(json.dumps(info,indent=2)); print(json.dumps(info))
