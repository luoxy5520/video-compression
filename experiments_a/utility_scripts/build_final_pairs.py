"""Build portable intermediate-feature -> solver-final target pairs."""
import argparse,json,hashlib
from pathlib import Path

def build(root):
    rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    records=[]
    for r in rows:
        sid=r['sample_id'];step=r['step'];base=Path(sid)
        latent=base/f'step_{step:03d}'/'clean_latent.pt'
        hidden=base/f'step_{step:03d}'/'hidden_cond_last_block.pt'
        raw=base/'final_raw.yuv'
        for x in [latent,hidden,raw,base/'final_latent.pt']:
            if not (root/x).is_file():raise FileNotFoundError(root/x)
        assert (root/raw).stat().st_size==10183680
        records.append(dict(sample_id=sid,step=step,total_steps=20,prompt=r['prompt'],seed=r['seed'],latent=str(latent),hidden=str(hidden),target_raw_yuv=str(raw),target_final_latent=str(base/'final_latent.pt'),target_label=str(base/'state.npz'),target_metadata=str(base/'metadata.json'),target_raw_sha256=hashlib.sha256((root/raw).read_bytes()).hexdigest()))
    assert len({(r['sample_id'],r['step']) for r in records})==len(records)
    for sid in {r['sample_id'] for r in records}:
        group=[r for r in records if r['sample_id']==sid]
        assert sorted(r['step'] for r in group)==[5,10,15,20]
        assert len({r['target_raw_sha256'] for r in group})==1
    obj=dict(schema_version='a2-final-v1',feature_root='.',label_root='SET_TO_ORACLE_LABEL_DIRECTORY',target='solver_final_compression_state',label_status='pending_external_oracle',notes=['latent/hidden/target_raw_yuv/target_final_latent are relative to this manifest directory.','target_label/target_metadata are relative to configurable labels_root.','All steps of a sample share ONE final label. Intermediate raw videos are not supervision targets.','Do not split steps of the same sample or seeds of the same prompt across partitions.'],video=dict(width=832,height=480,frames=17,fps=16,pixel_format='yuv420p',color_matrix='bt709',color_range='limited'),records=records)
    (root/'a2_final_pairs.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    print(f'PASS: {len(records)} feature records, {len(set(r["sample_id"] for r in records))} unique final targets')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args();build(Path(a.data_root))
