"""Verify sample identity, candidate mapping and historical metrics without training."""
import csv
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"P2_OnPolicy/scripts"))
from o0 import historical,p,strict
from o1 import infer,level_bounds
from l0 import HIST,chosen_rows,rank,step
from repair import to_pixels

ON=p.ROOT/"P2_OnPolicy"


def read(path):
    return json.loads(path.read_text())


class AuditDataset(Dataset):
    def __init__(self, rows):
        self.rows=rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        key,row=self.rows[index]
        sid=row["sample_id"]
        return (key,sid,torch.load(HIST/"features"/f"{sid}.pt",weights_only=True),
                torch.from_numpy(p.p05.load_npz(ON/"o0"/sid/"T0.npz")["q"]),
                torch.from_numpy(p.p05.load_npz(HIST/"labels"/f"{sid}.npz")["mode"]==1))


def tensors_equal(x,y):
    return all(torch.equal(a,b) for a,b in zip(x,y,strict=True))


def units():
    q=np.array([True,False,True,False])
    assert rank(np.array([4,2,3,1]),q,2)["recall"]==1
    assert rank(np.array([-1,-3,-2,-4]),q,2)["recall"]==1
    assert rank(np.array([1,4,2,3]),q,2)["recall"]==0
    legal=np.array([True,True,False,True,False])
    assert rank(np.array([2,3,100,-1,200])[legal],np.array([True,True,False,False,False])[legal],2)["recall"]==1
    net=p.SparseHead(67,16).cuda()
    x=torch.zeros((1,67,16,30,52),device="cuda")
    mask=torch.zeros((16,30,52),device="cuda",dtype=torch.bool)
    mask[:,0,0]=True
    target=torch.zeros((16,16,16),device="cuda",dtype=torch.long)
    target[:,0,0]=3
    captured=[]
    def retain(_,args,out):
        out.retain_grad()
        captured.append(out)
    hook=net.body.register_forward_hook(retain)
    pos,value,_=step(net,(x,mask,target),True,False)
    (pos+value).backward()
    hook.remove()
    spatial=captured[0].grad[0].movedim(0,-1)
    assert torch.count_nonzero(spatial[~mask])==0 and torch.count_nonzero(spatial[mask])>0
    assert net.select.weight.grad.norm()>0 and net.value.weight.grad.norm()>0
    return dict(perfect=True,negative_logits=True,reversed=True,illegal_rank_mask=True,masked_loss_gradient=True)


@torch.inference_mode()
def history(rows, manifest):
    output=[]
    for path in sorted(ON.glob("o1-*/run.json"))+sorted(ON.glob("fit-*/run.json")):
        r=read(path)
        name=path.parent.name
        assert p.digest(path)==manifest["runs"][name]["run_sha256"]
        assert p.digest(path.with_name("head.pt"))==manifest["runs"][name]["head_sha256"]
        scored=r["train"] if r["scope"]=="full" else r["points"]
        net=p.SparseHead(67,16).cuda().eval()
        net.load_state_dict(torch.load(path.with_name("head.pt"),weights_only=True))
        hits=[]
        for row in scored:
            sid=row["sample"]
            sample=rows[sid]
            x=torch.load(HIST/"features"/f"{sid}.pt",weights_only=True).cuda()
            mode=torch.from_numpy(p.p05.load_npz(HIST/"labels"/f"{sid}.npz")["mode"]==1).cuda()
            q=p.p05.load_npz(ON/"o0"/sid/f'{r["arm"]}.npz')["q"]
            out,*_=infer(net,x,mode,level_bounds("cuda"))
            actual=int(((q!=0)&(out!=0)).sum())
            assert actual==row["realized_support_overlap"],(name,sid,actual,row["realized_support_overlap"])
            hits.append(actual)
        assert {row["positive"] for row in scored}=={64}
        prompts={rows[row["sample"]]["prompt"] for row in scored}
        balanced=mean(mean(hit/64 for row,hit in zip(scored,hits) if rows[row["sample"]]["prompt"]==pr) for pr in prompts)
        output.append(dict(run=name,updates=r["updates"],videos=len(scored),hits=sum(hits),positives=64*len(scored),
                           video_macro_recall=mean(hits)/64,prompt_macro_recall=balanced,
                           min_hits=min(hits),max_hits=max(hits)))
    return output


def main():
    out=Path(sys.argv[1])
    assert (out/"audit/audit.json").is_file()
    rows=p.rows_by_split()
    assert len({r["prompt"] for r in rows["train"]})==11
    assert len({r["prompt"] for r in rows["val"]})==4
    assert not {r["prompt"] for r in rows["train"]}&{r["prompt"] for r in rows["val"]}
    assert len({r["sample_id"] for grp in rows.values() for r in grp})==60
    g=read(p.ROOT/"P2_InfoProbe/g_labels/manifest.json")
    hashes={x["sample"]:x for x in g}
    o0={r["sample"]:r for r in read(ON/"o0/metrics.json")}
    records=[]
    all_rows={r["sample_id"]:r for grp in rows.values() for r in grp}
    cases=chosen_rows(rows["train"])[:5]
    frozen=p.load_model(p.CKPT).eval().requires_grad_(False)
    fresh=[]
    for split,group in rows.items():
        for row in group:
            sid=row["sample_id"]
            q0=o0[sid]
            assert q0["split"]==split and q0["prompt"]==row["prompt"]
            f=HIST/"features"/f"{sid}.pt"
            assert p.digest(f)==hashes[sid]["feature_sha256"]
            root,source,base,fixed,refs=historical(split,row)
            assert p.digest(source)==q0["source_sha256"]
            assert np.array_equal(base["mode_id"],p.p05.load_npz(HIST/"labels"/f"{sid}.npz")["mode"])
            for arm in ("T0","T1"):
                qpath=ON/"o0"/sid/f"{arm}.npz"
                meta=q0["arms"][arm]
                assert p.digest(qpath)==meta["label_sha256"]
                assert meta["source_reference_stream_sha256"]==refs[arm][1]["stream_sha256"]
                assert p.digest(root/"fixed_mode_mv.npz")==meta["mode_mv_sha256"]
                q=p.p05.load_npz(qpath)["q"]
                assert q.shape==(*p.SHAPE,16) and not q[base["mode_id"]==0].any()
                assert np.all(np.count_nonzero(q,axis=(1,2,3,4))==4)
                assert np.all(np.count_nonzero(q,axis=-1)<=1)
                records.append(dict(sample=sid,split=split,prompt=row["prompt"],arm=arm,step=row["step"],
                    source_sha256=q0["source_sha256"],feature_sha256=hashes[sid]["feature_sha256"],
                    label_sha256=meta["label_sha256"],reference_sha256=meta["source_reference_stream_sha256"],
                    positives=64,candidates=q0["candidates"],strict_roundtrip=row in cases))
                if row in cases:
                    with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="m1-") as temp:
                        strict(Path(temp)/"jm",row,dict(base,residual_y=to_pixels(q)),source,fixed,q,mc_check=True)
            if row in cases:
                x=p.feature(frozen,p.input_tensor(row,"hidden"),base).cpu()
                torch.testing.assert_close(x,torch.load(f,weights_only=True),rtol=0,atol=0)
                fresh.append(sid)
    assert len({v["source_sha256"] for v in records if v["split"]=="train"})==44
    assert not {v["source_sha256"] for v in records if v["split"]=="train"}&{v["source_sha256"] for v in records if v["split"]=="val"}
    dataset=AuditDataset([(r["sample_id"],r) for r in cases])
    expected={k:(sid,*v) for k,sid,*v in dataset}
    checks=0
    for batch in (1,2,4):
        for workers in (0,2):
            for shuffle in (False,True):
                for keys,sids,x,q,mask in DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,
                                                      multiprocessing_context="spawn" if workers else None):
                    for k,s,fx,l,m in zip(keys,sids,x,q,mask,strict=True):
                        known=expected[k]
                        assert known[0]==s and tensors_equal(known[1:],(fx,l,m))
                        checks+=1
    for size in (2,4,8):
        copy=AuditDataset([(str(i),cases[0]) for i in range(size)])
        original=copy[0]
        for keys,sids,x,q,mask in DataLoader(copy,batch_size=2,shuffle=True,num_workers=0):
            for s,fx,l,m in zip(sids,x,q,mask,strict=True):
                assert s==original[1] and tensors_equal(original[2:],(fx,l,m))
    parent=Path(__file__).resolve().parents[2]
    res=dict(pass_m1=True,fresh_hidden_matches=fresh,dataloader_observations=checks,units=units(),
             original_metrics=history(all_rows,read(parent/"P2_OnPolicy/runs_manifest.json")))
    with (out/"m1_data_audit.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    p.save_json(out/"m1_unit_tests.json",res)
    print(res,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
