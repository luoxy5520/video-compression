"""Linux equivalent of environment/smoke/full/ideal Windows wrappers. Original parsers unchanged."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY=sys.executable
ENC=ROOT/'linux/JM/bin/lencod.exe';DEC=ROOT/'linux/JM/bin/ldecod.exe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def run(args,cwd,log,env=None):
    with Path(log).open('w') as f:subprocess.run([str(x) for x in args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
def encoder(folder,mode,state,raw,bit,recon):
    env=os.environ.copy();env.update(JM_STATE_MODE=mode,JM_STATE_FILE=str(state))
    params=dict(InputFile=raw,OutputFile=bit,ReconFile=recon,SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,RDOptimization=0,NumberReferenceFrames=1,PList0References=1)
    args=[ENC,'-d',ROOT/'JM-backend/cfg/encoder_baseline.cfg','-f',ROOT/'linux/skip_p16.cfg']
    for k,v in params.items():args+=['-p',f'{k} = {v}']
    run(args,folder,folder/f'{mode}.log',env)
def decode(folder,bit,prefix):
    jm=folder/f'{prefix}_jm.yuv';ff=folder/f'{prefix}_ffmpeg.yuv'
    run([DEC,'-d',ROOT/'JM-backend/cfg/decoder.cfg','-p',f'InputFile = {bit}','-p',f'OutputFile = {jm}','-p','RefFile = ""'],folder,folder/f'{prefix}_jm.log')
    run(['ffmpeg','-v','error','-y','-i',bit,'-pix_fmt','yuv420p','-f','rawvideo',ff],folder,folder/f'{prefix}_ffmpeg.log')
    assert jm.stat().st_size==ff.stat().st_size==10183680
    return jm,ff

def export(r,out):
    folder=out/'oracle_exports'/r['sample_id']/f'step_{r["step"]:03d}'
    if (folder/'metadata.json').exists():
        c=json.loads((folder/'closure_result.json').read_text())
        assert all(c[k] for k in ['bitstream_bit_exact','jm_decoded_pixels_exact','ffmpeg_decoded_pixels_exact']);return
    if folder.exists():folder.rename(folder.with_name(folder.name+'_incomplete_'+str(time.time_ns())))
    folder.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data'/r['raw_yuv'];assert raw.stat().st_size==10183680
    shutil.copy2(raw,folder/'input_yuv420p.yuv')
    state=folder/'oracle_state.bin';ref=folder/'reference.264';replay=folder/'replayed.264';recon=folder/'reference_recon.yuv'
    encoder(folder,'dump',state,raw,ref,recon)
    dummy=folder/'dummy_zero.yuv'
    with dummy.open('wb') as f:f.truncate(10183680)
    encoder(folder,'replay',state,dummy,replay,folder/'replay_encoder_recon.yuv')
    jm,ff=decode(folder,replay,'replay')
    c=dict(bitstream_bit_exact=sha(ref)==sha(replay),jm_decoded_pixels_exact=sha(recon)==sha(jm),ffmpeg_decoded_pixels_exact=sha(recon)==sha(ff))
    write(folder/'closure_result.json',c)
    assert all(c.values()),c
    run([PY,ROOT/'scripts/parse_b1_state.py','--input',state,'--output-npz',folder/'oracle_labels.npz','--metadata-json',folder/'state_metadata.json'],folder,folder/'parse.log')
    pairing=dict(sample_id=r['sample_id'],diffusion_step=str(r['step']),prompt=r['prompt'],seed=r['seed'],latent_ref=r['latent'],hidden_ref=r['hidden'],final_latent_ref=r['final_latent'],timestep=r['timestep'],sigma=r['sigma'],hidden_branch=r['hidden_branch'],hidden_block=r['hidden_block'],sampler=r['sampler'],total_steps=r['total_steps'])
    write(folder/'metadata.json',dict(pairing=pairing,input=dict(path=str(raw),sha256=sha(raw),width=832,height=480,frames=17,fps=16,color_matrix='bt709',color_range='limited'),h264=dict(mode_preset='skip-p16',qp=28,reference_frames=1,rd_optimization=0),validation=c,tools=dict(encoder_sha256=sha(ENC),decoder_sha256=sha(DEC))))
    print('EXPORT PASS',r['sample_id'],r['step'],c,flush=True)
def build(out):
    run([PY,ROOT/'scripts/build_dataset.py','--exports-root',out/'oracle_exports','--output-dir',out/'training_dataset','--feature-root',ROOT/'data','--feature-kind','both','--require-features','--group-by','prompt','--expected-mode-preset','skip-p16'],ROOT,out/'dataset_build.log')
    print((out/'training_dataset/dataset_summary.json').read_text(),flush=True)
def ideal(sid):
    import numpy as np
    base=ROOT/'output';folder=ROOT/'chain_tests/ideal'/sid/time.strftime('%Y%m%d-%H%M%S');folder.mkdir(parents=True)
    rows=[json.loads(x) for x in (base/'training_dataset/manifests/all.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['sample_id']==sid and str(r['step'])=='20')
    label=base/'training_dataset'/row['target_label_path'];template=Path(row['target_oracle_state_path'])
    state=folder/'predicted_state.bin'
    run([PY,ROOT/'scripts/prediction_to_b1_state.py','--prediction',label,'--oracle-template',template,'--output-state',state,'--result-json',folder/'adapter_result.json'],folder,folder/'adapter.log')
    dummy=folder/'dummy_zero.yuv'
    with dummy.open('wb') as f:f.truncate(10183680)
    bit=folder/'predicted.264';encoder(folder,'replay',state,dummy,bit,folder/'predicted_encoder_recon.yuv')
    jm,ff=decode(folder,bit,'predicted')
    oracle=base/'oracle_exports'/sid/'step_020'/'reference.264'
    result=dict(sample_id=sid,validation=dict(jm_matches_ffmpeg=sha(jm)==sha(ff),standard_decode_valid=sha(jm)==sha(ff),ideal_matches_oracle_bitstream=sha(bit)==sha(oracle)),bitstream_bytes=bit.stat().st_size,oracle_i_frame=True)
    write(folder/'prediction_replay_result.json',result);assert all(result['validation'].values()),result
    print('IDEAL PASS',result,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','one','all','ideal']);a=p.parse_args()
    cfg=(ROOT/'JM-backend/cfg/state_ip_p16.cfg').read_text().replace('PSliceSkip               = 0','PSliceSkip               = 1');(ROOT/'linux/skip_p16.cfg').write_text(cfg)
    if a.action=='check':
        import numpy,torch
        assert ENC.exists() and DEC.exists();assert shutil.which('ffmpeg') and shutil.which('ffprobe')
        for folder in sorted((ROOT/'data').glob('sample_*')):
            x=torch.load(folder/'step_020/clean_latent.pt',weights_only=True);y=torch.load(folder/'final_latent.pt',weights_only=True)
            assert torch.equal(x,y),folder
            final=folder/'final_raw.yuv'
            if final.exists():assert sha(final)==sha(folder/'step_020/raw.yuv'),folder
        write(ROOT/'linux/environment_check.json',dict(python=sys.version,numpy=numpy.__version__,step20_equals_final=True,encoder_sha256=sha(ENC),decoder_sha256=sha(DEC)))
        print('CHECK PASS: dependencies; all four step20 latents/raw equal final outputs',flush=True);return
    if a.action=='ideal':
        for sid in ['sample_0000','sample_0001','sample_0002','sample_0003']:ideal(sid)
        return
    rows=json.loads((ROOT/'data/paired_data.local.json').read_text())['records']
    if a.action=='one':rows=[next(r for r in rows if r['sample_id']=='sample_0000' and r['step']==20)]
    out=ROOT/('output_smoke' if a.action=='one' else 'output');out.mkdir(exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r: export(r,out),rows))
    build(out)
if __name__=='__main__':main()
