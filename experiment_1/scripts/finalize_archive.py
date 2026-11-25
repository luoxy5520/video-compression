"""Audit final artifacts and record source/archive provenance without rerunning models."""
import argparse
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path
import runtime as r
from run import ROOT,E0,save

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--code-commit',required=True);args=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.code_commit)
    expected={'E1_0/rows.json':144,'E1_1/sensitivity.json':882,'E1_3/skip_probe_results.json':816,'latency_breakdown.json':12,'video_manifest.json':29}
    for name,count in expected.items():assert len(json.loads((ROOT/'results'/name).read_text()))==count,(name,count)
    mixed=json.loads((ROOT/'results/E1_2/mixed_precision_results.json').read_text())
    policies=json.loads((ROOT/'configs/frozen_mixed_policies.json').read_text())['policies']
    validation=[v for v in mixed if v['split']=='validation']
    assert len(validation)==128 and all(v['policy'] in policies for v in validation)
    assert len({v['sample'] for v in validation})==16
    with (ROOT/'results/actual_bitstream_breakdown.csv').open() as f:
        for row in csv.DictReader(f):
            total=sum(int(row[k]) for k in ('quantization_json_value_bytes','auxiliary_json_value_bytes','other_header_bytes','compressed_payload_bytes'))
            assert total==int(row['total_bytes'])==(ROOT/row['stream']).stat().st_size
    with (ROOT/'results/E1_3/block_residual_statistics.csv').open() as f:count=sum(1 for _ in csv.DictReader(f))
    assert count==597168
    manifest=json.loads((E0/'results/validation_manifest.json').read_text())
    assert len(manifest)==16 and all(v['split']=='val' for v in manifest)
    sources=[(s['sample'],s['latent'],'development') for s in r.samples]+[(s['sample_id'],s['final_latent'],'validation') for s in manifest]
    inputs=[]
    for name,path,split in sources:
        tensor=r.torch.load(path,weights_only=True,map_location='cpu')
        inputs.append(dict(sample=name,split=split,path=path,file_bytes=Path(path).stat().st_size,file_sha256=sha(Path(path)),tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),shape=list(tensor.shape),dtype=str(tensor.dtype),reference_yuv_path=str(E0/'results'/name/'reference.yuv'),reference_yuv_sha256=sha(E0/'results'/name/'reference.yuv')))
    save(ROOT/'results/input_manifest.json',inputs)
    sources={str(path.relative_to(ROOT)):sha(path) for path in ROOT.rglob('*.py') if '__pycache__' not in path.parts}
    save(ROOT/'results/source_sha256_final_analysis.json',sources)
    save(ROOT/'results/archive_receipt.json',dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),archive_source_commit=args.code_commit,execution_versions='Earlier actual source hashes retained separately; archive commit is not claimed as original execution commit',remote='git@github.com:luoxy5520/video-compression.git',branch='codex/experiment-1-latent-rd',server_root=str(ROOT),uniform=144,sensitivity=882,mixed=len(mixed),validation_mixed=128,reuse=816,block_statistics=count,latency_representatives=12,videos=29))
    inventory=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_inventory.json':inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    save(ROOT/'results/artifact_inventory.json',inventory)
    print('Archive verified:',len(inventory),'files;',count,'block rows; 29 videos; all byte decompositions exact')

if __name__=='__main__':main()
