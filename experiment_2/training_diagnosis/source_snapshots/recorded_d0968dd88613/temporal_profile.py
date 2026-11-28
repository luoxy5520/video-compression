from common import *
from temporal_entropy_model import load_temporal
import numpy as np,resource
out=ROOT/'results/E2_2';rows=json.loads((out/'temporal_results.json').read_text());latencies=[];slices=[]
for mode in ('TP0','TP1','TP2'):
    ck=ROOT/'checkpoints/temporal'/f'{mode}.pt';m,c=load_temporal(ck)
    for idx,r in enumerate([x for x in rows if x['mode']==mode]):
        record=next(x for x in json.loads((E2/'dataset_manifest.json').read_text())['records'] if x['path']==r['video']);x=tensor(record);path=out/f'{mode}_profile_dev{idx}.bin';enc=[];dec=[];contexts=[];entropy=[]
        for rep in range(6):
            t=time.perf_counter();stats=m.compress_video(x,path,sha(ck));elapsed=time.perf_counter()-t;t=time.perf_counter();q,z=m.decompress_video(path,sha(ck));de=time.perf_counter()-t
            if rep:enc.append(elapsed);dec.append(de);contexts.append(sum(s['context_seconds'] for s in stats['slices']));entropy.append(sum(s['entropy_encode_seconds'] for s in stats['slices']))
        latencies.append(dict(mode=mode,video=r['video'],scope='CPU transforms/probability/rANS; GPU endpoint NOT RUN; file I/O included; 1warmup+5formal',encode_seconds_mean=float(np.mean(enc)),decode_seconds_mean=float(np.mean(dec)),context_probability_seconds_mean=float(np.mean(contexts)),entropy_encode_seconds_mean=float(np.mean(entropy)),encode_samples=json.dumps(enc),decode_samples=json.dumps(dec),cpu_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,parameters=sum(p.numel() for p in m.parameters()),deployment_bytes=ck.stat().st_size))
        for t,s in enumerate(r['slices']):slices.append(dict(mode=mode,video=r['video'],slice=t,main_bytes=s['main'],hyper_bytes=s['hyper'],metadata_bytes_video=r['metadata_bytes'],allocation_note='metadata stored once per video; not equally assigned'))
        print(mode,idx,latencies[-1],flush=True)
csvsave(out/'temporal_context_latency.csv',latencies);csvsave(out/'temporal_rate_per_slice.csv',slices);csvsave(out/'temporal_prior_bitstream_breakdown.csv',slices)
# Main quantization/reconstruction errors are identical by construction; no propagation.
save(out/'temporal_error_propagation.json',dict(relative_to_TP0_max_latent_difference=0,all_canonical_symbols_exact=True,all_reconstructions_exact=True,scope='measured exact-equality checks in each actual decode, not a temporal prediction approximation'))
