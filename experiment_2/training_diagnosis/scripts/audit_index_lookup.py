"""CPU real-stream equality and standard independent decoder on all18 development clips."""
from common import *
from temporal_entropy_model import load_temporal
from index_lookup import install_bucketize
import subprocess
ck=ROOT/'checkpoints/extensions/P2_D42_joint10000/P2.pt';m,c=load_temporal(ck);digest=sha(ck)
records=[r for r in json.loads((E2/'dataset_manifest.json').read_text())['records'] if r['role']=='historical_development' and r['exists']];assert len(records)==18
out=ROOT/'results/index_lookup_audit';out.mkdir(exist_ok=True);rows=[]
standard=m.base.gc.build_indexes
with torch.no_grad():
    for idx,r in enumerate(records):
        x=tensor(r);fixed=m.fixed_sequence(x);baseline=out/f'{idx:02d}_standard.bin';candidate=out/f'{idx:02d}_bucketize.bin'
        m.base.gc.build_indexes=standard;a=m.compress_fixed(fixed,list(x.shape[-2:]),baseline,digest)
        install_bucketize(m);b=m.compress_fixed(fixed,list(x.shape[-2:]),candidate,digest)
        assert baseline.read_bytes()==candidate.read_bytes()
        q,z=m.decompress_video(candidate,digest);target=out/f'{idx:02d}_independent.pt'
        subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(candidate),str(target)],check=True)
        decoded=torch.load(target,weights_only=True);assert torch.equal(decoded['q'],q) and torch.equal(decoded['z'],z)
        rows.append(dict(video=r['path'],actual_bytes=b['actual_bytes'],stream_sha256=b['sha256'],bitstream_identical=True,standard_independent_decoder=True,latent_exact=True))
        save(out/'rows.json',rows);print(idx,b['actual_bytes'],'identical, standard decoder PASS',flush=True)
        m.base.gc.build_indexes=standard
save(out/'status.json',dict(status='PASS',historical_videos=18,checkpoint_sha256=digest,source_sha256=sha(ROOT/'scripts/index_lookup.py'),all_bitstreams_identical=True,all_standard_independent_decode=True,scope='CPU-only equality audit; no encoder latency inference; no sealed test access'))
