from run_followup import *
import torch
# Policy fixed before examining new encoding outcomes. No validation access.
policy=dict(native_stop_multiplier=2,dit_gated='same multiplier only if cached latent seed improves live initial cost',seed_count=1,QP=28,search_mode=3)
write(OUT/'stage-d3/policy_frozen.json',policy)
rows=[]
for s in (DEV[0],DEV[6]):
 # Warmup + three read/normalize/match/write measurements, includes feature I/O.
 sys.path.insert(0,str(OLD));import probe
 p=probe.DATA/probe.SETS[int(int(s.split('_')[1])>=1000)]/s/'final_latent.pt'
 times=[]
 for i in range(4):
  t=time.perf_counter();z=probe.tensor(p).numpy().reshape(16,5,30,2,52,2).transpose(1,2,4,3,5,0).reshape(5,30,52,64)
  z=probe.normalization(z);c=(16*np.stack([probe.match(z[j],z[j-1]) for j in range(1,5)])).repeat(4,axis=0).astype(np.int16)
  c.tofile(OUT/'stage-d1'/f'{s}_candidates.bin');times.append(time.perf_counter()-t)
 write(OUT/'stage-d3'/f'{s}_feature_times.json',dict(seconds=times,median_s=statistics.median(times[1:]),includes='feature read, normalization, matching, candidate file write; excludes Python interpreter/module startup'))
 for arm,mode in [('B1-measured',0),('Native-budget',1),('DiT-seed',2),('DiT-gated',3)]:
  r=encode(s,arm,ROOT/'jm-d3/bin/lencod.exe',envextra=dict(LMP_MODE=str(mode),LMP_CANDIDATES=str(OUT/'stage-d1'/f'{s}_candidates.bin')))
  if mode==0:
   assert r['sha256']==json.loads((OUT/'stage-d0'/s/'B1/result.json').read_text())['sha256']
  r['feature_s']=statistics.median(times[1:]) if mode>=2 else 0;r['pipeline_wall_s']=r['JM_wall_s']+r['feature_s'];rows.append(r)
write(OUT/'stage-d3/d3_results.json',rows)
