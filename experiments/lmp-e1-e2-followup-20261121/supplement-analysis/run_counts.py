import sys,os,json,time,csv
from pathlib import Path
import numpy as np
R=Path.home()/'lmp-e1-e2-followup-20261121';O=R/'supplement-analysis';sys.path[:0]=[str(R),str(O)];from run_followup import DEV,rawold,tools,kit,sha,command,csvout,write
SAMPLES=DEV+['sample_1028','sample_1029'];rows=[];checks=[]
# Fixed old policies; one diagnostic run per arm, no speed conclusion from logged runs.
for s in SAMPLES:
 for arm,mode in [('B1',0),('Native-budget',1),('DiT-seed',2),('DiT-gated',3)]:
  folder=O/'encodes'/s/arm;folder.mkdir(parents=True,exist_ok=True);raw=rawold(s)/'input_yuv420p.yuv';env={k:v for k,v in os.environ.items() if not k.startswith('JM_')};env.update(LMP_MODE=str(mode),LMP_CANDIDATES=str(O/f'{s}_candidates.bin'),LMP_BLOCK_CSV=str(folder/'blocks.csv'))
  def encode(enc,prefix,trace):
   bit=folder/f'{prefix}.264';recon=folder/f'{prefix}_recon.yuv';params=dict(InputFile=raw,OutputFile=bit,ReconFile=recon,SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,RDOptimization=0,NumberReferenceFrames=1,PList0References=1,SearchMode=3)
   args=[enc,'-d',kit/'JM-backend/cfg/encoder_baseline.cfg','-f',kit/'linux/skip_p16.cfg']
   for k,v in params.items():args+=['-p',f'{k} = {v}']
   e=env.copy()
   if not trace:e.pop('LMP_BLOCK_CSV',None)
   command(args,folder,folder/f'{prefix}.log',e);return bit,recon
  bit,recon=encode(R/'jm-perblock/bin/lencod.exe','instrumented',True)
  control,control_recon=encode(R/'jm-d3/bin/lencod.exe','frozen_control',False);assert sha(bit)==sha(control) and sha(recon)==sha(control_recon),(s,arm,'changed behavior')
  if mode==0:
   original,original_recon=encode(tools['ENC'],'original_B1',False);assert sha(bit)==sha(original) and sha(recon)==sha(original_recon)
  jm,ff=tools['decode'](folder,bit,'verified');assert sha(jm)==sha(ff)==sha(recon)
  a=np.genfromtxt(folder/'blocks.csv',delimiter=',',names=True,dtype=None,encoding=None);assert len(a)==24960
  assert np.array_equal(np.sort((a['frame']-1)*1560+(a['y']//16)*52+a['x']//16),np.arange(24960))
  prof=next(json.loads(x[12:]) for x in (folder/'instrumented.log').read_text().splitlines() if x.startswith('LMP_PROFILE '))
  assert int(a['integer_evals'].sum())==prof['integer_evals'] and int(a['subpixel_evals'].sum())==prof['subpixel_evals'],(s,arm,'counter mismatch')
  # Same inputs for exact counter repeat on two prior pilot samples, all four arms.
  repeated=False
  if s in (DEV[0],DEV[6]):
   before=(folder/'blocks.csv').read_bytes();bit2,recon2=encode(R/'jm-perblock/bin/lencod.exe','counter_repeat',True);assert sha(bit2)==sha(bit) and before==(folder/'blocks.csv').read_bytes();repeated=True
  native=int(a['integer_evals'].sum()-a['seed_evals'].sum());seed=int(a['seed_evals'].sum());sub=int(a['subpixel_evals'].sum())
  rows.append(dict(sample=s,split='dev' if s in DEV else 'historical_val_diagnostic',arm=arm,blocks=len(a),native_integer_evals=native,seed_evals=seed,total_integer_evals=native+seed,subpixel_evals=sub,total_evals=native+seed+sub,seed_duplicate_initial=int(a['seed_duplicate_initial'].sum()),seed_improved=int(a['seed_improved'].sum()),threshold_raised=int(a['threshold_raised'].sum()),initial_threshold_stop=int((a['exit_reason']==1).sum()),derived_threshold_stop=int((a['exit_reason']==2).sum()),predictor_threshold_stop=int((a['exit_reason']==5).sum()),pattern_completed=int((a['exit_reason']==6).sum()),seed_not_better=int((a['seed_status']==3).sum()),out_of_range=int((a['seed_status']==2).sum()),candidate_unavailable=int((a['seed_status']==1).sum()),threshold_effective_initial=int(((a['predictors_entered']==0)&(a['threshold_raised']==1)&(a['after_seed_cost']>a['initial_stop'])).sum())))
  checks.append(dict(sample=s,arm=arm,bitstream_sha256=sha(bit),recon_sha256=sha(recon),jm_sha256=sha(jm),ffmpeg_sha256=sha(ff),frozen_behavior_equal=True,original_B1_equal=mode==0,counter_sum_matches_global=True,counter_repeat_exact=repeated))
  write(folder/'correctness.json',checks[-1]);csvout(O/'search_totals_partial.csv',rows);print(s,arm,native,seed,sub,flush=True)
csvout(O/'search_totals.csv',rows);write(O/'correctness.json',checks)
