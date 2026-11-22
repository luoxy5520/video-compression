import os,sys,json,time,hashlib,statistics,shutil,subprocess,runpy,csv,platform
from pathlib import Path
import numpy as np
ROOT=Path.home()/'lmp-e1-e2-followup-20261121'
OLD=Path.home()/'lmp-h264-prior.uFe5ZV'
OUT=ROOT/'repo/results-e1-e2-followup';OUT.mkdir(exist_ok=True)
for d in ('stage-d0','stage-d1','stage-d2','stage-d3'): (OUT/d).mkdir(exist_ok=True)
sys.path.insert(0,str(OLD));from baseline import KITS,kit,tools,sha
DEV=['sample_0000','sample_0004','sample_0008','sample_0012','sample_0016','sample_1000','sample_1004','sample_1008']
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def csvout(p,rows):
 if not rows:return
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def command(args,cwd,log,env=None):
 with (OUT/'commands.log').open('a') as f:f.write(json.dumps([str(x) for x in args])+'\n')
 tools['run'](args,cwd,log,env)
def rawold(s):return KITS[int(int(s.split('_')[1])>=1000)]/'output/oracle_exports'/s/'step_020'
def quality(raw,recon):
 a=np.fromfile(raw,np.uint8).reshape(17,-1)[:,:832*480].astype(float);b=np.fromfile(recon,np.uint8).reshape(17,-1)[:,:832*480].astype(float)
 mse=((a-b)**2).mean(axis=1)
 return [float(10*np.log10(255**2/mse[x].mean())) for x in (slice(1,None),slice(-4,None))]
def encode(s,arm,enc,extra=None,envextra=None):
 folder=OUT/('stage-d0' if arm in ('B1','profile') else 'stage-d3')/s/arm;folder.mkdir(parents=True,exist_ok=True)
 bit=folder/'stream.264';recon=folder/'recon.yuv';raw=rawold(s)/'input_yuv420p.yuv'
 params=dict(InputFile=raw,OutputFile=bit,ReconFile=recon,SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,RDOptimization=0,NumberReferenceFrames=1,PList0References=1,SearchMode=3)
 params.update(extra or {});args=[enc,'-d',kit/'JM-backend/cfg/encoder_baseline.cfg','-f',kit/'linux/skip_p16.cfg']
 for k,v in params.items():args+=['-p',f'{k} = {v}']
 env={k:v for k,v in os.environ.items() if not k.startswith('JM_')};env.update(envextra or {})
 times=[];hashes=[];profiles=[]
 for i in range(4):
  t=time.perf_counter();command(args,folder,folder/f'encode{i}.log',env);times.append(time.perf_counter()-t);hashes.append(sha(bit))
  lines=(folder/f'encode{i}.log').read_text().splitlines();profiles.append([json.loads(x[12:]) for x in lines if x.startswith('LMP_PROFILE ')])
 assert len(set(hashes))==1
 jm,ff=tools['decode'](folder,bit,'verified');assert sha(jm)==sha(ff)==sha(recon)
 q=quality(raw,recon);r=dict(sample=s,arm=arm,bitrate_kbps=bit.stat().st_size*8/(17/16)/1000,P_Y_PSNR=q[0],last4_Y_PSNR=q[1],JM_wall_s=statistics.median(times[1:]),min_s=min(times[1:]),max_s=max(times[1:]),seconds=times,sha256=hashes[0],dual_decode=True,profiles=profiles)
 write(folder/'result.json',r);print(json.dumps(r),flush=True);return r
if __name__=='__main__':
 os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 stage=sys.argv[1]
 if stage=='audit':
  write(OUT/'manifest.json',dict(dev=DEV,validation_policy='historically observed; not used for tuning',base_commit='f080a451831e89a59948006c4299bca698fdfef6',scope='E1 to E2; cached full VAE source only',remote=str(ROOT),model_directory='/mnt/data_hdd/models'))
  (OUT/'env.txt').write_text(subprocess.check_output('uname -a; lscpu; uptime; free -h; git status --short; git rev-parse HEAD',shell=True,cwd=ROOT/'repo',text=True))
  rows=[]
  for line in (OLD/'audit.jsonl').read_text().splitlines():
   a=json.loads(line)
   if a['sample'] not in DEV:continue
   for key,hkey in [('raw',None),('latent','latent_sha256'),('hidden','hidden_sha256'),('labels','labels_sha256')]:
    expected=a['input']['sha256'] if key=='raw' else a[hkey];actual=sha(a[key]);assert actual==expected,(a['sample'],key)
    rows.append(dict(sample=a['sample'],kind=key,path=a[key],sha256=actual,match=True))
  write(OUT/'stage-d0/input_audit.json',rows)
 elif stage=='baseline':
  rs=[encode(s,'B1',tools['ENC']) for s in (DEV[0],DEV[6])];old=[json.loads(x) for x in (OLD/'baseline.log').read_text().splitlines()]
  for r in rs:
   expected=next(x for x in old if x['sample']==r['sample'] and x['mode']==3);assert r['sha256']==expected['sha256']
  csvout(OUT/'stage-d0/d0_baseline.csv',[{k:v for k,v in r.items() if k not in ('profiles','seconds')} for r in rs])
 elif stage=='profile':
  for s in (DEV[0],DEV[6]):
   r=encode(s,'profile',ROOT/'jm-profile/bin/lencod.exe');b=json.loads((OUT/'stage-d0'/s/'B1/result.json').read_text());assert r['sha256']==b['sha256']
