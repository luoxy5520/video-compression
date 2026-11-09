"""Locate first persistent scale collapse and probe gradient conflict."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import torch

from m1 import ON,p
from o1 import data_for
from train import sample_rows,training


def read(path):
    return json.loads(path.read_text())


def curves(root):
    def read_curve(n,seed):
        a=read(root/f"scale-{n}-{seed}/run.json")
        return mean(r["recall"] for r in a["snapshots"][str(a["steps"])]["train"])
    n=next((n for n in (2,4,8,44) if all(read_curve(n,z)<.8 for z in (1701,1702))),None)
    data=dict(first_persistent_collapse=n,max_train_recall={str(n):{str(z):read_curve(n,z) for z in (1701,1702)}
           for n in (1,2,4,8,44)})
    p.save_json(root/"m2_gate.json",data)
    return n


def gradient(root,n,seed):
    rows=sample_rows(p.rows_by_split()["train"],n)[:4]
    data=[data_for(ON/"o0",r,"T0") for r in rows]
    net=p.SparseHead(67,16).cuda()
    torch.manual_seed(seed)
    net=p.SparseHead(67,16).cuda()
    directory=root/f"scale-{n}-{seed}"
    results=[]
    end=read(directory/"run.json")["steps"]
    for checkpoint in sorted({0,400,end}):
        if checkpoint:
            net.load_state_dict(torch.load(directory/f"head-{checkpoint}.pt",weights_only=True))
        gradients=[]
        for row,x in zip(rows,data,strict=True):
            net.zero_grad(set_to_none=True)
            loss,detail=training(net,x,"hard",collect=True)
            loss.backward()
            shared=torch.cat([q.grad.detach().flatten().clone() for q in net.body.parameters()])
            selection=torch.cat([q.grad.detach().flatten().clone() for q in net.select.parameters()])
            gradients.append((row["sample_id"],shared,selection))
        for (a,x1,s1),(b,x2,s2) in itertools.combinations(gradients,2):
            cs=torch.nn.functional.cosine_similarity
            results.append(dict(seed=seed,checkpoint=checkpoint,sample_a=a,sample_b=b,
                    shared_cosine=float(cs(x1,x2,dim=0)),select_cosine=float(cs(s1,s2,dim=0)),
                    shared_norm_a=float(x1.norm()),shared_norm_b=float(x2.norm()),
                    select_norm_a=float(s1.norm()),select_norm_b=float(s2.norm())))
    p.save_json(root/f"m2-gradient-{seed}.json",results)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--seed",type=int)
    a=cli.parse_args()
    torch.set_num_threads(2)
    if a.seed:
        n=read(a.root/"m2_gate.json")["first_persistent_collapse"]
        gradient(a.root,n,a.seed)
    else:
        print(curves(a.root) or 0)


if __name__=="__main__":
    main()
