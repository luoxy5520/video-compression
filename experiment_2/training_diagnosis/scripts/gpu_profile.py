"""GPU-start hybrid deployment: GPU analysis/h_a, CPU probability/rANS/decoder.
Probability/CDF computation stays on CPU at both ends. No encoder VAE call.
Full H.264 measurements rerun from GPU latent, not assembled historical times.
"""
from pixels import *
from temporal_entropy_model import load_temporal
import copy,resource,math,argparse
parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['TP2','P2','P3'],default='TP2');parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output-name');parser.add_argument('--bucketize-indexes',action='store_true');args=parser.parse_args()
import fcntl
gpu_lock=(ROOT/'logs/gpu0_exclusive.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
out=ROOT/'results/performance'/(args.output_name or args.method);out.mkdir(parents=True,exist_ok=True);ck=args.checkpoint or ROOT/'checkpoints'/('temporal/TP2.pt' if args.method=='TP2' else 'temporal_joint/P2.pt' if args.method=='P2' else 'temporal_pixel/P3.pt');cpu,c=load_temporal(ck);gpu=copy.deepcopy(cpu.base).cuda().eval();v=vae();records=json.loads((E2/'dataset_manifest.json').read_text())['records'];base=json.loads((ROOT/'results/E2_1d/D16_f1_seed20261127/development_step10000.json').read_text());rows=[]
if args.bucketize_indexes:
    from index_lookup import install_bucketize
    install_bucketize(cpu)
save(out/'runtime_policy.json',dict(index_lookup='bucketize' if args.bucketize_indexes else 'CompressAI1.2.8',checkpoint_sha256=sha(ck),profiling_script_sha256=sha(__file__),scope='runtime index mapping only; unchanged model weights and format'))
def measured(fn):
    torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-t

def encode_hybrid(original,path):
    times={};t=time.perf_counter();ys,times['analysis_gpu_seconds']=measured(lambda:gpu.ga(original.permute(1,0,2,3).contiguous()));zs,times['hyper_analysis_gpu_seconds']=measured(lambda:gpu.ha(ys));(yc,zc),times['D2H_seconds']=measured(lambda:(ys.cpu(),zs.cpu()));tq=time.perf_counter();qs=yc.round();times['main_quantization_cpu_seconds']=time.perf_counter()-tq;fixed=[];th=time.perf_counter()
    for i in range(len(qs)):
        zstream=cpu.base.eb.compress(zc[i:i+1]);zh=cpu.base.eb.decompress(zstream,zc.shape[-2:]);hs=cpu.base.hs(zh)[:,:,:qs.shape[-2],:qs.shape[-1]]+.11;fixed.append((qs[i:i+1],hs,zstream[0],list(zc.shape[-2:])))
    times['hyper_entropy_and_synthesis_cpu_seconds']=time.perf_counter()-th;stats=cpu.compress_fixed(fixed,list(qs.shape[-2:]),path,sha(ck));times.update(context_probability_seconds=sum(s['context_seconds'] for s in stats['slices']),index_seconds=sum(s['index_seconds'] for s in stats['slices']),entropy_main_seconds=sum(s['entropy_encode_seconds'] for s in stats['slices']),reference_update_seconds=sum(s['reference_update_seconds'] for s in stats['slices']));times['full_gpu_latent_to_bitstream_seconds']=time.perf_counter()-t
    return stats,qs,times

def h264_encode(original,path,qp):
    # End-to-end actual source GPU latent -> native VAE -> RGB/YUV -> H.264.
    image=rgb(v.decode([original])[0]);raw=yuv(image)
    subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-preset','medium','-qp',str(qp),'-g','17','-bf','2','-refs','3','-threads','4','-color_primaries','bt709','-colorspace','bt709','-color_trc','bt709','-color_range','tv','-f','h264',str(path)],input=raw,check=True)
    return raw

def h264_decode(path):return subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(path),'-f','rawvideo','-pix_fmt','yuv420p','pipe:1'],capture_output=True,check=True).stdout

def psnr_y_bytes(a,b):
    ay=np.frombuffer(a,dtype=np.uint8).reshape(17,-1)[:,:832*480].astype(float);by=np.frombuffer(b,dtype=np.uint8).reshape(17,-1)[:,:832*480].astype(float);return float(10*np.log10(255**2/max(((ay-by)**2).mean(),1e-30)))
for idx,source in enumerate(base):
    record=next(r for r in records if r['path']==source['video']);x=tensor(record);original=x.permute(1,0,2,3).contiguous().cuda();path=out/f'gpu_dev{idx}.bin';samples=[]
    with torch.no_grad():
        ref=rgb(v.decode([original])[0]);reference_yuv=yuv(ref)
        for rep in range(6):
            torch.cuda.reset_peak_memory_stats();stats,expected,enc=encode_hybrid(original,path);t=time.perf_counter();decode_profile={};q,z=cpu.decompress_video(path,sha(ck),decode_profile);decode_seconds=time.perf_counter()-t;assert torch.equal(q,expected);(image,vae_seconds)=measured(lambda:rgb(decode(v,z)));ty=time.perf_counter();receiver_yuv=yuv(image);yuv_seconds=time.perf_counter()-ty
            if rep:samples.append(dict(**enc,**decode_profile,entropy_and_synthesis_decode_cpu_seconds=decode_seconds,receiver_vae_and_RGB_seconds=vae_seconds,receiver_RGB_to_YUV_seconds=yuv_seconds,post_dit_two_ends_seconds=enc['full_gpu_latent_to_bitstream_seconds']+decode_seconds+vae_seconds+yuv_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        save(out/f'codec_timing_dev{idx}.json',samples)
        quality_row=quality(ref,image,reference_yuv);cpuq=torch.cat([f[0] for f in cpu.fixed_sequence(x)]);qdiff=int((cpuq!=q).sum());external=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(path),str(external)],check=True);assert torch.equal(torch.load(external,weights_only=True)['z'],z)
        # Actual per-video QP calibration near this deployment's quality, then5formal reruns.
        candidates=[];hpath=out/f'dev{idx}.264'
        for qp in (18,22,26,28,32,37,42):
            raw=h264_encode(original,hpath,qp);decoded=h264_decode(hpath);candidates.append(dict(qp=qp,psnr_y=psnr_y_bytes(reference_yuv,decoded),bytes=hpath.stat().st_size))
        # Refine between neighboring QPs rather than call sparse points equal-quality.
        best=min(candidates,key=lambda r:abs(r['psnr_y']-quality_row['psnr_y']))
        for qp in range(max(1,best['qp']-2),min(51,best['qp']+2)+1):
            if qp in [r['qp'] for r in candidates]:continue
            raw=h264_encode(original,hpath,qp);decoded=h264_decode(hpath);candidates.append(dict(qp=qp,psnr_y=psnr_y_bytes(reference_yuv,decoded),bytes=hpath.stat().st_size))
        best=min(candidates,key=lambda r:abs(r['psnr_y']-quality_row['psnr_y']));hsamples=[]
        for rep in range(6):
            _,enc_seconds=measured(lambda:h264_encode(original,hpath,best['qp']));t=time.perf_counter();decoded=h264_decode(hpath);dec_seconds=time.perf_counter()-t
            if rep:hsamples.append(dict(gpu_latent_to_h264_seconds=enc_seconds,h264_decode_seconds=dec_seconds,two_ends_seconds=enc_seconds+dec_seconds))
        save(out/f'h264_timing_dev{idx}.json',hsamples)
        equal_quality=bool(abs(best['psnr_y']-quality_row['psnr_y'])<=.5)
        rows.append(dict(video=record['path'],actual_bytes=stats['actual_bytes'],bpp=stats['actual_bytes']*8/(832*480*17),kbps=stats['actual_bytes']*8/1000/(17/16),quality=quality_row,main_coding_symbols_changed_cpu_vs_gpu=qdiff,gpu_analysis_cpu_probability=True,independent_decode=True,codec_samples=samples,h264_candidates=candidates,h264_selected=best,h264_samples=hsamples,quality_tolerance_db=.5,within_quality_tolerance=equal_quality,encoder_speed_ratio=(float(np.mean([s['gpu_latent_to_h264_seconds'] for s in hsamples]))/float(np.mean([s['full_gpu_latent_to_bitstream_seconds'] for s in samples]))) if equal_quality else 'NOT COMPARABLE',codec_to_h264_byte_ratio=stats['actual_bytes']/best['bytes'],parameters=sum(p.numel() for p in cpu.parameters()),weight_bytes=ck.stat().st_size,cpu_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='1warmup+5formal, actual GPU-start encode; active other GPU services present; no Prompt/DiT timing; both receiver endpoints are rawYUV420; VAE receiver includes latentH2D, RGB transfer and RGB-to-YUV conversion'))
        save(out/'deployment_benchmark.json',rows);print(json.dumps(rows[-1]),flush=True)
