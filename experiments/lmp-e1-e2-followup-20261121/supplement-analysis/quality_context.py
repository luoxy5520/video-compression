import sys,json,csv
from pathlib import Path
import numpy as np
R=Path.home()/'lmp-e1-e2-followup-20261121';O=R/'supplement-analysis';sys.path.insert(0,str(R));from run_followup import DEV,rawold,csvout
rows=[]
for s in DEV+['sample_1028','sample_1029']:
 source=np.fromfile(rawold(s)/'input_yuv420p.yuv',np.uint8).reshape(17,-1)[:,:480*832]
 for arm in ('B1','Native-budget','DiT-seed','DiT-gated'):
  f=O/'encodes'/s/arm;recon=np.fromfile(f/'instrumented_recon.yuv',np.uint8).reshape(17,-1)[:,:480*832];mse=[]
  for i in range(17):
   diff=source[i].astype(np.int16)-recon[i].astype(np.int16);mse.append(float(np.square(diff.astype(np.int32)).mean()))
  p=10*np.log10(255**2/np.mean(mse[1:]));last=10*np.log10(255**2/np.mean(mse[-4:]));rate=(f/'instrumented.264').stat().st_size*8/(17/16)/1000
  r=dict(sample=s,arm=arm,P_Y_PSNR=p,last4_Y_PSNR=last,bitrate_kbps=rate)
  if arm=='B1':base=r
  r.update(P_Y_PSNR_delta=p-base['P_Y_PSNR'],last4_delta=last-base['last4_Y_PSNR'],bitrate_delta_pct=(rate/base['bitrate_kbps']-1)*100);r['quality_pass']=r['P_Y_PSNR_delta']>=-.1 and r['last4_delta']>=-.1 and r['bitrate_delta_pct']<=3;rows.append(r)
csvout(O/'quality_context.csv',rows)
