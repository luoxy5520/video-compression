"""Wait for workers, then isolated performance, plots, reports and artifact audit."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def snapshot(label):
    values={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')}
    (ROOT/'results'/f'source_sha256_{label}.json').write_text(json.dumps(values,indent=2))

def run(command,log,env):
    with (ROOT/'logs/postprocess_commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),command=command))+'\n')
    with (ROOT/'logs'/log).open('a') as f:subprocess.run(command,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pipeline-pid',type=int,required=True);p.add_argument('--profiling-codec',required=True);args=p.parse_args()
    snapshot('quality_pipeline')
    while not (ROOT/'results/ALL_MAIN_MEASUREMENTS_COMPLETE').exists():
        try:
            os.kill(args.pipeline_pid,0)
            if Path(f'/proc/{args.pipeline_pid}/stat').read_text().split()[2]=='Z':raise ProcessLookupError
        except ProcessLookupError:raise RuntimeError('quality pipeline exited before completion; inspect pipeline.log')
        time.sleep(15)
    assert len(json.loads((ROOT/'results/E1_0/rows.json').read_text()))==144
    assert len(json.loads((ROOT/'results/E1_1/sensitivity.json').read_text()))==882
    assert len(json.loads((ROOT/'results/E1_3/skip_probe_results.json').read_text()))==816
    # Optional profiling instrumentation serializes identical bytes, tested below and per artifact.
    shutil.copy2(args.profiling_codec,ROOT/'codec/lvc.py');snapshot('latency_and_delivery')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    run([sys.executable,'-m','unittest','discover','-s','tests','-v'],'tests_final.log',env)
    for name in ('measure_latency','summarize','extra_figures','videos','report'):
        task_env=dict(env,PYTHONPATH=str(ROOT.parent/'tools_plot')) if name in ('summarize','extra_figures') else env
        run([sys.executable,'-u',str(ROOT/'scripts'/f'{name}.py')],name+'_final.log',task_env)
    required=['final_report.md','results/latency_breakdown.json','results/actual_bitstream_breakdown.csv','results/E1_1/channel_rank_stability.csv','results/E1_2/oracle_gap.csv','results/E1_3/energy_distortion_correlation.csv','results/video_manifest.json']
    assert all((ROOT/f).exists() for f in required)
    assert len(json.loads((ROOT/'results/video_manifest.json').read_text()))==29
    # Main stage measurements and specified analyses complete; optional items remain explicitly NOT RUN.
    status=json.loads((ROOT/'experiment_status.json').read_text())
    for name in ('E1_0','E1_1','E1_2','E1_3'):status[name]='COMPLETE'
    status['delivery']='COMPLETE';status['optional_not_run']=['LPIPS: package/weights unavailable','optional quantization range ablation','BD-Rate: measured-range interpolation supplied instead','actual Skip codec: controlled perturbation route used','validation Oracle: development-only restricted greedy search']
    (ROOT/'experiment_status.json').write_text(json.dumps(status,indent=2))
    (ROOT/'results/DELIVERY_COMPLETE').touch()
    inventory=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ('artifact_inventory.json','postprocess.log'):
            inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT/'results/artifact_inventory.json').write_text(json.dumps(inventory,indent=2))
    print('DELIVERY_COMPLETE',flush=True)

if __name__=='__main__':main()
