from pathlib import Path
import json,re,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'output';dataset=out/'training_dataset'
rows=[json.loads(x) for x in (dataset/'manifests/all.jsonl').read_text().splitlines()];assert len(rows)==16
exports=[]
for p in sorted((out/'oracle_exports').glob('*/step_*/closure_result.json')):
    c=json.loads(p.read_text());assert all(c[k] for k in ['bitstream_bit_exact','jm_decoded_pixels_exact','ffmpeg_decoded_pixels_exact'])
    log=(p.parent/'replay.log').read_text();m=re.search(r'Total ME time for sequence\s*:\s*([0-9.]+)',log);assert m and float(m[1])==0
    exports.append(dict(sample_id=p.parent.parent.name,step=p.parent.name,validation=c,replay_me_seconds=float(m[1])))
assert len(exports)==16
for sid in {r['sample_id'] for r in rows}:
    group=[r for r in rows if r['sample_id']==sid];assert len({r['target_label_path'] for r in group})==1
for r in rows:
    assert Path(r['latent_path']).exists() and Path(r['hidden_path']).exists()
    with np.load(dataset/r['target_label_path']) as a:
        assert a['mode_id'].shape==(16,30,52) and np.isin(a['mode_id'],[0,1]).all()
ideal=[]
for sid in sorted({r['sample_id'] for r in rows}):
    p=sorted((ROOT/'chain_tests/ideal'/sid).glob('*/prediction_replay_result.json'))[-1];r=json.loads(p.read_text());assert all(r['validation'].values());ideal.append(r)
pilot=dataset/'manifests/pilot_prompt_split';pilot.mkdir(exist_ok=True)
for name,ids in [('train',{'sample_0000','sample_0001'}),('test',{'sample_0002','sample_0003'})]:
    (pilot/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['sample_id'] in ids))
report=dict(status='PASS',environment=json.loads((ROOT/'linux/environment_check.json').read_text()),record_count=16,unique_final_targets=4,rejected_records=0,oracle_exports=exports,ideal_chain=ideal,default_split=json.loads((dataset/'dataset_summary.json').read_text())['counts']['splits'],pilot_split=dict(train=8,test=8,val=0,note='Separate non-overwriting prompt split; only pipeline pilot, no validation set.'),prediction_test='NOT RUN: no trained full mode/MV/residual predictions',training='NOT RUN',oracle_i_dependency=True)
(ROOT/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('AUDIT PASS: 16 closures + zero replay ME + 4 ideal replays + grouped final targets')
