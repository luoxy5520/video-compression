"""Finish the 32-video experiment after generation; checkpointed phase status."""
from pathlib import Path
import os,json,sys,time,subprocess,shutil
REPO=Path('/home/luoxiangyu/dit-paired-experiment');BASE=REPO/'expansion_20261106';ROOT=Path('/home/luoxiangyu/senior-oracle-expand32-20261106');SOURCE=Path('/home/luoxiangyu/senior-oracle-dataset-kit-20261106');DATA=Path('/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106');PY=sys.executable
status={}
def phase(name,state):
    status.update(phase=name,state=state,updated=time.strftime('%Y-%m-%d %H:%M:%S'));(BASE/'pipeline32_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
def cmd(args,log,env=None):
    with Path(log).open('w') as f:subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
def main():
    phase('generation','waiting')
    while len(list(DATA.glob('sample_*/complete.json')))<32:
        if not Path('/proc/'+sys.argv[1]).exists():raise RuntimeError('Generation exited before 32 completed samples; inspect generation32.log')
        time.sleep(5)
    rows=[json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines()];assert len(rows)==128
    for r in rows:
        for k in ['latent','hidden','raw_yuv','final_latent']:r[k]=str(Path(r[k]).relative_to(DATA))
    (DATA/'paired_data.local.json').write_text(json.dumps(dict(records=rows),indent=2))
    ROOT.mkdir(exist_ok=True);(ROOT/'linux').mkdir(exist_ok=True)
    for dest,source in [(ROOT/'data',DATA),(ROOT/'JM-backend',SOURCE/'JM-backend'),(ROOT/'scripts',SOURCE/'scripts'),(ROOT/'linux/JM',SOURCE/'linux/JM')]:
        if not dest.exists():dest.symlink_to(source,target_is_directory=True)
    shutil.copy2(SOURCE/'linux/run_pipeline.py',ROOT/'linux/run_pipeline.py')
    runner=ROOT/'linux/run_pipeline.py'
    phase('environment','running');cmd([PY,runner,'check'],BASE/'check32.log')
    phase('oracle_128_records','running');cmd([PY,runner,'all'],BASE/'oracle32.log')
    dataset=ROOT/'output/training_dataset';accepted=json.loads((dataset/'dataset_summary.json').read_text())['counts'];assert accepted['accepted_records']==128 and accepted['rejected_records']==0
    # Group split from generation plan, independent of package's hash-based default.
    split=json.loads((BASE/'splits32.json').read_text());records=[json.loads(x) for x in (dataset/'manifests/all.jsonl').read_text().splitlines()]
    target=dataset/'manifests/explicit_prompt_split';target.mkdir(exist_ok=True)
    for name,ids in split.items():
        selected=[dict(r,split=name) for r in records if r['sample_id'] in ids];(target/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    phase('a0_a2_mode_training','running');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['OMP_NUM_THREADS']='4'
    cmd([PY,BASE/'train_mode.py','--config',BASE/'config_train32.json'],BASE/'train32.log',env)
    phase('complete','PASS');status['dataset']=str(dataset);status['training']=json.loads((BASE/'config_train32.json').read_text())['output'];(BASE/'pipeline32_status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as e:
        phase('failed',repr(e));raise
