"""Create a compact shareable HTML table from actual training metrics."""
import argparse,json,html
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('metrics');a=p.parse_args();path=Path(a.metrics);m=json.loads(path.read_text());rows=[]
for name,group in [('A0 final-only',m['a0_test']),('A2 shared-time',m['test_by_step'])]:
    for s,r in group.items():
        b=r['fixed_skip_baseline'];rows.append([name,s,*[f'{r[k]:.4f}' for k in ['accuracy','macro_f1','skip_recall','p16_recall']],f'{b["accuracy"]:.4f}',f'{b["macro_f1"]:.4f}'])
heads=['model','step','accuracy','Macro-F1','Skip recall','P16 recall','fixed Skip accuracy','fixed Skip F1']
s='<html><meta charset="utf-8"><title>Oracle 模式预测实验</title><style>body{font:16px sans-serif;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:9px}th{background:#eee}</style><h1>真实 Oracle 模式预测实验</h1><p>源数据：'+html.escape(str(path))+'</p><p>样本数 '+str(m['sample_count'])+'；参数量 '+str(m['parameters'])+'。仅模式预测，不含MV/residual。所有步骤监督目标均为最终P帧状态。</p><table><tr>'+''.join('<th>'+h+'</th>' for h in heads)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table><p>少量prompt的pilot，不能作为泛化或在线提前退出的证据。置信度未经校准。</p><pre>'+html.escape(json.dumps(m['config']['splits'],indent=2))+'</pre></html>'
path.with_name('report.html').write_text(s);print(path.with_name('report.html'))
from pathlib import Path
import shutil,json
repo=Path('/home/luoxiangyu/dit-paired-experiment/expansion_20261106');run=Path('/mnt/data_hdd/luoxiangyu/dit_pairs/training_expand32_20261106');dest=repo/'results_expand32';dest.mkdir(exist_ok=True)
for p in run.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
kit=Path('/home/luoxiangyu/senior-oracle-expand32-20261106');idx=repo/'dataset_index32';idx.mkdir(exist_ok=True)
shutil.copy2(kit/'output/training_dataset/dataset_summary.json',idx/'dataset_summary.json')
shutil.copytree(kit/'output/training_dataset/manifests/explicit_prompt_split',idx/'explicit_prompt_split',dirs_exist_ok=True)
shutil.copy2(Path('/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106/training_index.json'),idx/'training_index.json')
backup=Path('/mnt/data_hdd/luoxiangyu/oracle_experiments/expand32_20261106');backup.mkdir(parents=True,exist_ok=True)
shutil.copytree(kit/'output',backup/'output',dirs_exist_ok=True)
shutil.copy2(repo/'audit32.json',backup/'audit32.json')
print('Saved trained checkpoints/predictions/metrics/index; backed up all expanded oracle artifacts')
