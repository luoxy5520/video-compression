"""Checkpointed orchestration, with frozen development policies before validation."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from run import ROOT,save
from analyze import table

def launch(script,args,gpu,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*args]
    with (ROOT/'logs/pipeline_commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),command=command,gpu=gpu))+'\n')
    log=(ROOT/'logs'/f'{name}.log').open('w')
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    log.close();return process

def join(processes):
    codes=[p.wait() for p in processes]
    if any(codes):raise RuntimeError(f'workers failed: {codes}; inspect logs')

def state(stage,measurement=None):
    p=ROOT/'experiment_status.json';s=json.loads(p.read_text());s[stage]='RUNNING'
    if measurement:s[stage+'_measurements']=measurement
    save(p,s)

def freeze_mixed(rows):
    uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
    policies=json.loads((ROOT/'configs/mixed_development_grid.json').read_text());scores=[]
    for policy in policies:
        candidates=[v for v in rows if v['policy']==policy]
        assert len(candidates)==2
        gains=[]
        for v in candidates:
            base=min([u for u in uniform if u['sample']==v['sample'] and u['method']!='Q0'],key=lambda u:abs(np.log(u['total_bytes']/v['total_bytes'])))
            gains.append(v['psnr_y']-base['psnr_y'])
        scores.append(float(np.mean(gains)))
    selected=[]
    for method in ('M1','M2','M2_three','M3'):
        choices=sorted([i for i,p in enumerate(policies) if p['method']==method],key=lambda i:scores[i],reverse=True)[:2]
        selected.extend(policies[i] for i in choices)
    save(ROOT/'configs/frozen_mixed_policies.json',dict(source_split='development only',selection='nearest actual-rate screening; final comparisons use measured-range interpolation',policies=selected))

def main():
    p=argparse.ArgumentParser();p.add_argument('--existing-sensitivity-pid',type=int);args=p.parse_args()
    try:
        if not (ROOT/'results/SENSITIVITY_MEASUREMENTS_COMPLETE').exists():
            state('E1_1')
            if args.existing_sensitivity_pid:
                while not (ROOT/'configs/frozen_channel_ranking.json').exists():
                    os.kill(args.existing_sensitivity_pid,0);time.sleep(5)
                cmd=Path(f'/proc/{args.existing_sensitivity_pid}/cmdline').read_bytes()
                assert b'scripts/analyze.py' in cmd and b'sensitivity' in cmd and b'--worker' not in cmd
                os.kill(args.existing_sensitivity_pid,signal.SIGTERM)
            rows=[r for r in json.loads((ROOT/'results/E1_1/sensitivity.json').read_text()) if r['split']=='development']
            assert len(rows)==98
            join([launch('analyze.py',['--stage','sensitivity','--worker',str(i),'--workers','4'],i%2,f'sensitivity_worker_{i}') for i in range(4)])
            for i in range(4):rows+=json.loads((ROOT/f'results/E1_1/sensitivity_worker_{i}.json').read_text())
            assert len(rows)==882
            save(ROOT/'results/E1_1/sensitivity.json',rows)
            table(ROOT/'results/E1_1/channel_sensitivity.csv',rows)
            (ROOT/'results/SENSITIVITY_MEASUREMENTS_COMPLETE').touch();state('E1_1','COMPLETE')
        if not (ROOT/'results/MIXED_MEASUREMENTS_COMPLETE').exists():
            state('E1_2')
            join([launch('mixed.py',['--subset','development','--worker',str(i),'--workers','2'],i,f'mixed_development_{i}') for i in range(2)])
            rows=[]
            for i in range(2):rows+=json.loads((ROOT/f'results/E1_2/mixed_precision_results_development_{i}.json').read_text())
            freeze_mixed(rows)
            join([launch('mixed.py',['--subset','validation','--worker',str(i),'--workers','4'],i%2,f'mixed_validation_{i}') for i in range(4)])
            validation=[]
            for i in range(4):validation+=json.loads((ROOT/f'results/E1_2/mixed_precision_results_validation_{i}.json').read_text())
            assert len(validation)==128
            save(ROOT/'results/E1_2/mixed_precision_results.json',rows+validation)
            table(ROOT/'results/E1_2/mixed_precision_results.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows+validation])
            (ROOT/'results/MIXED_MEASUREMENTS_COMPLETE').touch();state('E1_2','COMPLETE')
        if not (ROOT/'results/REUSE_MEASUREMENTS_COMPLETE').exists():
            state('E1_3')
            join([launch('reuse.py',['--worker',str(i),'--workers','4'],i%2,f'reuse_worker_{i}') for i in range(4)])
            rows=[]
            for i in range(4):rows+=json.loads((ROOT/f'results/E1_3/skip_probe_results_worker_{i}.json').read_text())
            assert len(rows)==816
            save(ROOT/'results/E1_3/skip_probe_results.json',rows)
            table(ROOT/'results/E1_3/skip_probe_results.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
            (ROOT/'results/REUSE_MEASUREMENTS_COMPLETE').touch();state('E1_3','COMPLETE')
        (ROOT/'results/ALL_MAIN_MEASUREMENTS_COMPLETE').touch()
    except Exception as error:
        save(ROOT/'results/pipeline_failure.json',dict(error=repr(error)))
        raise

if __name__=='__main__':main()
