"""Fail-closed E1 baseline gate and fixed uniform scan. Run on 113Server."""
import argparse
import csv
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
import runtime as r
from codec.lvc import encode, decode

ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT.parent
MODEL = Path('/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def independent(path):
    target = path.with_suffix('.npy')
    subprocess.run([sys.executable, str(ROOT/'scripts/decode.py'), str(path), str(target)], check=True)
    value = np.load(target)
    target.unlink()
    return value

def ssim(a, b, frames, width, height):
    args = ['ffmpeg','-v','info']
    for path in (a,b):
        args += ['-f','rawvideo','-pix_fmt','yuv420p','-s',f'{width}x{height}','-r','16','-i',str(path)]
    proc = subprocess.run(args+['-lavfi','ssim','-f','null','-'], capture_output=True, check=True)
    import re
    return float(re.findall(r'All:([0-9.]+)',proc.stderr.decode())[-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['gate','uniform'], required=True)
    args = parser.parse_args()
    status_path = ROOT/'experiment_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    stage = 'Stage0' if args.stage=='gate' else 'E1_0'
    status[stage] = 'RUNNING'; save(status_path,status)
    try:
        if args.stage!='gate':
            assert json.loads((ROOT/'results/baseline_gate.json').read_text())['status']=='PASSED'
        assert digest(MODEL)=='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'
        manifest = json.loads((E0/'results/validation_manifest.json').read_text())
        assert len(manifest)==16 and all(m['split']=='val' for m in manifest)
        r.vae = r.WanVAE(vae_pth=str(MODEL), device='cuda:0')
        sources = [(s['sample'],s['latent'],'development',s) for s in r.samples]
        if args.stage=='uniform':
            sources += [(s['sample_id'],s['final_latent'],'validation',s) for s in manifest]
        old_rows = json.loads((E0/'results/rows.json').read_text())
        rows = []; gate = []
        save(ROOT/'results/environment.json',dict(python=sys.version,platform=platform.platform(),torch=r.torch.__version__,cuda=r.torch.version.cuda,model_sha256=digest(MODEL)))
        for name, latent_path, split, source in sources:
            folder = ROOT/'results'/stage/name; folder.mkdir(parents=True,exist_ok=True)
            ref = json.loads((E0/'results'/name/'reference.json').read_text())
            expected = source.get('latent_sha256') if split=='development' else None
            file_sha = digest(Path(latent_path))
            if expected: assert file_sha==expected
            z = r.torch.load(latent_path,weights_only=True,map_location='cpu').numpy()[None]
            assert z.dtype==np.float32 and z.shape==(1,16,5,60,104)
            assert digest_bytes(z.tobytes()) == (ref.get('latent_sha256') or json.loads((E0/'results'/name/'statistics.json').read_text())['sha256'])
            r.pixels(z)  # GPU warmup
            (rgb,_,_,_), vt = r.timing(lambda:r.pixels(z))
            raw = r.yuv(rgb)
            assert raw==(E0/'results'/name/'reference.yuv').read_bytes(), 'VAE reference mismatch'
            reference = folder/'reference.yuv'; reference.write_bytes(raw)
            meta = dict(source,vae_sha256=digest(MODEL),normalization='solver final latent unchanged; native WanVAE.decode applies self.scale internally',width=832,height=480,frames=17,fps=16)
            for bits in ([2,4,8] if args.stage=='gate' else [0,2,3,4,5,6,7,8]):
                cfg = dict(bits=bits,granularity='channel')
                stream = folder/f'Q{bits}.lvc'
                sizes, et = r.timing(lambda:encode(z,stream,cfg,dict(algorithm='zstd'),meta))
                zh, dt = r.timing(lambda:decode(stream))
                assert np.array_equal(zh, independent(stream))
                if args.stage=='gate' or (split=='development' and bits in (4,6,8)):
                    (rec,_,_,_), rt = r.timing(lambda:r.pixels(zh))
                else:
                    rec,elapsed,_,_ = r.pixels(zh)
                    rt = dict(single=elapsed, note='representative repeated VAE timings: development Q4/Q6/Q8')
                rr = r.yuv(rec); reconstructed = folder/'reconstruction.yuv'; reconstructed.write_bytes(rr)
                metrics = r.quality(rgb,rec,raw,rr)
                row = dict(sample=name,split=split,method=f'Q{bits}',**sizes,bitrate_kbps=stream.stat().st_size*8/(17/16)/1000,compression_ratio=z.nbytes/stream.stat().st_size,encode_s=et,decode_s=dt,vae_s=rt,reference_vae_s=vt,ssim=ssim(reference,reconstructed,17,832,480),**metrics)
                if args.stage=='gate':
                    oldname={2:'D2',4:'C2',8:'B2'}[bits]
                    oldstream=E0/'results'/name/f'{oldname}.lvc'
                    assert np.array_equal(zh,decode(oldstream)), 'E0 latent baseline mismatch'
                    old=next(v for v in old_rows if v['sample']==name and v['method']==oldname)
                    assert abs(metrics['psnr_y']-old['psnr_y'])<1e-6, 'E0 pixel metric mismatch'
                    oldblob=oldstream.read_bytes(); n=struct.unpack('<I',oldblob[4:8])[0]; h=json.loads(oldblob[8:8+n])
                    encode(z,folder/'baseline_exact.lvc',h['quantization'],h['compression'],h['video'])
                    assert (folder/'baseline_exact.lvc').read_bytes()==oldblob, 'E0 bytes mismatch'
                    gate.append(dict(sample=name,bits=bits,latent_file_sha256=file_sha,reference_sha256=digest_bytes(raw),exact_bitstream=True,psnr_y=metrics['psnr_y']))
                rows.append(row); save(ROOT/'results'/stage/'rows.json',rows)
                print(name,bits,row['total_bytes'],row['psnr_y'],flush=True)
            reconstructed.unlink(); reference.unlink()
        if args.stage=='gate': save(ROOT/'results/baseline_gate.json',dict(status='PASSED',checks=gate))
        else:
            path=ROOT/'results/E1_0/uniform_quantization.csv'
            with path.open('w') as f:
                keys=['sample','split','method','total_bytes','header_bytes','payload_bytes','bitrate_kbps','compression_ratio','psnr_y','psnr_rgb','ssim','worst_frame_psnr_y','last4_psnr_y']
                writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        status[stage]='COMPLETE' if args.stage=='gate' else 'RUNNING';status[stage+'_measurements']='COMPLETE';save(status_path,status)
    except Exception as error:
        status[stage]='FAILED';status['last_error']=repr(error);save(status_path,status)
        raise

def digest_bytes(data): return hashlib.sha256(data).hexdigest()

if __name__=='__main__': main()
