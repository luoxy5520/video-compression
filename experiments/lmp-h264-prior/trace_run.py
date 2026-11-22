import os,sys,json
from pathlib import Path
from baseline import kit,KITS,run,sha
root=Path.cwd()
for sample in sys.argv[1:]:
    old=KITS[int(int(sample.split('_')[1])>=1000)]/'output/oracle_exports'/sample/'step_020'
    out=root/'trace'/sample
    out.mkdir(parents=True,exist_ok=True)
    bit=out/'stream.264'
    opts=dict(InputFile=old/'input_yuv420p.yuv',OutputFile=bit,
              ReconFile=out/'recon.yuv',SourceWidth=832,SourceHeight=480,
              FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,
              RDOptimization=0,NumberReferenceFrames=1,PList0References=1,SearchMode=0)
    cmd=[root/'jm-trace/bin/lencod.exe','-d',kit/'JM-backend/cfg/encoder_baseline.cfg',
         '-f',kit/'linux/skip_p16.cfg']
    for k,v in opts.items():
        cmd+=['-p',f'{k} = {v}']
    run(cmd,out,out/'encode.log',{k:v for k,v in os.environ.items() if not k.startswith('JM_')})
    assert sha(bit)==sha(old/'reference.264')
    trace=[line for line in (out/'encode.log').read_text().splitlines() if line.startswith('LMP_MVP ')]
    assert len(trace)==16*30*52,len(trace)
    (out/'mvp.tsv').write_text('\n'.join(x[8:] for x in trace)+'\n')
    print(json.dumps(dict(sample=sample,rows=len(trace),sha256=sha(bit))),flush=True)
