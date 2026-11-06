"""Report both fixed class baselines in addition to training-majority baseline."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('metrics');a=p.parse_args();path=Path(a.metrics);obj=json.loads(path.read_text())
def score(cm):
    total=sum(map(sum,cm));out={'accuracy':(cm[0][0]+cm[1][1])/total};f=[]
    for i in [0,1]:
        tp=cm[i][i];rec=tp/max(1,sum(cm[i]));pre=tp/max(1,cm[0][i]+cm[1][i]);f.append(2*rec*pre/max(1e-12,rec+pre))
    out['macro_f1']=sum(f)/2;return out
def walk(x):
    if isinstance(x,dict):
        if 'confusion_matrix' in x and 'majority_baseline' in x:
            c=x['confusion_matrix'];n0=sum(c[0]);n1=sum(c[1]);x['class_support']={'skip':n0,'p16':n1};x['fixed_skip_baseline']=score([[n0,0],[n1,0]]);x['fixed_p16_baseline']=score([[0,n0],[0,n1]])
        for v in list(x.values()):walk(v)
    elif isinstance(x,list):
        for v in x:walk(v)
walk(obj);path.write_text(json.dumps(obj,indent=2));print('Added fixed Skip and P16 baselines:',path)
