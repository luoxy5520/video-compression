from baseline import HOME,KITS,sha
from pathlib import Path
import json
repo=HOME/'dit-a1-three-state-20261109'
train=json.loads((repo/'expansion_20261106/config_train32.json').read_text())['splits']['train']
new=json.loads((repo/'a_state_20261107/new48.json').read_text())['samples']
train+= [x['sample_id'] for x in new if x['split']=='train']
val=json.loads((HOME/'a1_quality_20261108/A1_ThreeState/full/run_manifest.json').read_text())['videos']
assert (len(train),len(val))==(44,16)
assert not set(train).intersection(x['sample'] for x in val)
data=Path('/mnt/data_hdd/luoxiangyu/dit_pairs')
for sample in train+[x['sample'] for x in val]:
    new=int(sample.split('_')[1])>=1000
    pair=data/('exp_a_new48_20261107' if new else 'exp_expand32_20261106')/sample
    old=KITS[new]/'output/oracle_exports'/sample/'step_020'
    meta=json.loads((old/'metadata.json').read_text())
    assert meta['pairing']['sample_id']==sample and meta['pairing']['hidden_branch']=='conditional'
    raw=old/'input_yuv420p.yuv'
    assert sha(raw)==sha(pair/'final_raw.yuv')==sha(pair/'step_020/raw.yuv')==meta['input']['sha256']
    h=next((x for x in val if x['sample']==sample),None)
    hidden=pair/'step_020/hidden_cond_last_block.pt'
    labels=KITS[new]/'output/training_dataset/labels'/f'{sample}__step-20.npz'
    if h:
        assert (h['prompt'],h['seed'])==(meta['pairing']['prompt'],meta['pairing']['seed'])
        for path,key in [(hidden,'hidden'),(labels,'oracle_labels'),(raw,'source')]:
            assert sha(path)==h['input'][key+'_sha256'],(sample,key,path)
    print(json.dumps(dict(sample=sample,split='train' if h is None else 'val',pairing=meta['pairing'],
        input=meta['input'],h264=meta['h264'],raw=str(raw),hidden=str(hidden), labels=str(labels),
        latent=str(pair/'final_latent.pt'),baseline_sha256=sha(old/'reference.264'),
        hidden_sha256=sha(hidden),latent_sha256=sha(pair/'final_latent.pt'),labels_sha256=sha(labels))),flush=True)
