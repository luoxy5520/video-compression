import csv,json,sys
from pathlib import Path
import numpy as np
R=Path.home()/'lmp-e1-e2-followup-20261121';O=R/'supplement-analysis';sys.path.insert(0,str(R));from run_followup import DEV,csvout,write
samples=DEV+['sample_1028','sample_1029'];rows=[];pooled={}
for label,ss in [('dev_pooled',DEV)]+[(s,[s]) for s in samples]:
 a={k:np.concatenate([np.load(O/f'{s}_offline.npz')[k] for s in ss]) for k in ('truth','magnitude','phase','2x2_cosine_hit','2x2_cosine_epe','2x2_cosine_sad')};outside=(np.abs(a['truth'])>18*4).any(axis=1);miss=~a['2x2_cosine_hit'];rows.append(dict(sample=label,N=len(miss),outside_max_window=int(outside.sum()),outside_fraction=float(outside.mean()),latent_misses=int(miss.sum()),fraction_misses_outside_window=float(outside[miss].mean()),coverage_inside_window=float(a['2x2_cosine_hit'][~outside].mean()),oracle_all81_coverage_upper_bound=float((~outside).mean())))
 if label=='dev_pooled':pooled=a
csvout(O/'d2_search_support.csv',rows)
a=pooled;mbins=np.digitize(a['magnitude'],[1,4,16]);weights=np.bincount(mbins,minlength=4)/len(mbins);std=[]
for p in range(1,5):
 coverage=0;emean=0;sm=0
 for b in range(4):
  m=(a['phase']==p)&(mbins==b);assert m.any();coverage+=weights[b]*a['2x2_cosine_hit'][m].mean();emean+=weights[b]*a['2x2_cosine_epe'][m].mean();sm+=weights[b]*a['2x2_cosine_sad'][m].mean()
 m=a['phase']==p;std.append(dict(phase=p,N=int(m.sum()),coverage=float(a['2x2_cosine_hit'][m].mean()),motion_standardized_coverage=float(coverage),motion_standardized_epe_mean=float(emean),motion_standardized_sad_mean=float(sm)))
csvout(O/'d2_phase_standardized.csv',std);write(O/'d2_interpretation_data.json',dict(support=rows,phase=std))
