"""Aggregate per-video results without changing the locked run decisions."""
from statistics import mean

from common import *

def average(xs):
    return mean([x for x in xs if x is not None]) if any(x is not None for x in xs) else None

def paired(a,b):
    delta = [x-y for x,y in zip(a,b,strict=True) if x is not None and y is not None]
    return dict(delta=mean(delta),ci=bootstrap(delta),n=len(delta)) if delta else None

def gz0():
    records = []
    for path in sorted(OUT.glob("gz0-*-*-*/metrics.json")):
        row = read(path)
        keys = ("fold","arm","seed","params","elapsed_s","training_peak_bytes","checkpoint_sha256","groups")
        item = {k:row[k] for k in keys}
        item["gate"] = all(item["groups"][group][metric][0]>0 for group in ("B","D")
            for metric in ("recall_delta_prior_ci","recall_delta_random_ci","ap_delta_prior_ci","ap_delta_random_ci"))
        records.append(item)
    assert len(records)==8
    return records

def summarize_sparse(ids,rows,base):
    report = {}
    for method in ("linear","knn","prior"):
        results = [rows[s][method] for s in ids]
        report[method] = {metric:average([r[metric] for r in results]) for metric in ("ap","recall","precision")}
        report[method]["ap_vs_F0"] = paired([r["ap"] for r in results],[base[s][method]["ap"] for s in ids])
        report[method]["recall_vs_F0"] = paired([r["recall"] for r in results],[base[s][method]["recall"] for s in ids])
    report["sampled_opportunity"] = mean(rows[s]["sampled_opportunity"] for s in ids)
    report["coverage"] = mean(rows[s]["coverage"] for s in ids)
    for tag in ("knn_pair","random_pair"):
        report[tag] = {k:average([rows[s][tag][k] for s in ids])
            for k in ("pos_recall","precision","agreement","sign","level_mae","support_iou","both_positive")}
    report["top1_pos_recall_vs_F0"] = paired([rows[s]["knn_pair"]["pos_recall"] for s in ids],
                                            [base[s]["knn_pair"]["pos_recall"] for s in ids])
    return report

def summarize_continuous(ids,rows):
    report = {}
    metrics = ("spearman","pearson","energy_mae","log_energy_mae","top_q_recall","sign_outside_near_zero",
               "quantized_nonzero_disagree","quantized_nonzero_recall","quantized_threshold_sensitivity","near_zero_fraction")
    for method in ("linear","knn","prior","random","zero"):
        rs = [rows[s][method] for s in ids]
        report[method] = {m:average([r[m] for r in rs]) for m in metrics}
        report[method]["low_mae_by_frequency"] = np.mean([r["low_mae_by_frequency"] for r in rs],axis=0).tolist()
        if method in ("linear","knn"):
            for ref in ("prior","random","zero"):
                for m in ("log_energy_mae","top_q_recall"):
                    report[method][f"{m}_vs_{ref}"] = paired([r[m] for r in rs],[rows[s][ref][m] for s in ids])
                report[method][f"low_mae_vs_{ref}"] = [
                    paired([r["low_mae_by_frequency"][i] for r in rs],[rows[s][ref]["low_mae_by_frequency"][i] for s in ids])
                    for i in range(3)]
    return report

def stage(tag,folds):
    records = []
    doc = read(OUT/f"{tag}.json")
    for r in doc["results"]:
        scores = r["scores"]
        base = next(x for x in doc["results"] if x["fold"]==r["fold"] and x["feature"]=="F0")
        groups = {}
        for group in ("B","D"):
            ids = folds[r["fold"]][group]
            result = summarize_sparse(ids,{s:v["sparse"] for s,v in scores.items()},
                                      {s:v["sparse"] for s,v in base["scores"].items()})
            if tag=="gz2":
                result["continuous"] = summarize_continuous(ids,{s:v["continuous"] for s,v in scores.items()})
            groups[group] = result
        records.append({**{k:r[k] for k in ("fold","feature","aligned_dimension","common_dimension")},"groups":groups})
    return dict(seconds=doc["seconds"],records=records)

def main():
    folds = read(OUT/"split_manifest.json")["folds"]
    first = gz0()
    gz1,gz2 = (stage(s,folds) for s in ("gz1","gz2"))
    t = read(OUT/"gz2_labels.json")
    decision = dict(gz0=first,gz1=gz1,gz2=gz2,
        replay_pass=[a for a in ("A","C1") if all(r["gate"] for r in first if r["arm"]==a)],
        t0_t1={k:average([x[k] for x in t]) for k in ("t0_t1_iou","t0_t1_overlap","t0_t1_sign","t0_t1_level_mae")})
    p.save_json(HERE/"metrics/summary.json",decision)

if __name__=="__main__":
    main()
