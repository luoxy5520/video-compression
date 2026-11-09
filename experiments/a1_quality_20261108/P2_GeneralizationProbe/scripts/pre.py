"""Audit source identity and re-evaluate frozen historical heads."""
import hashlib
from statistics import mean

from common import *

def main():
    torch.set_num_threads(2)
    audit = read(OUT / "pre_a/audit.json")
    rows = p.rows_by_split()
    o0 = keyed(read(ON / "o0/metrics.json"))
    contract = keyed(read(HISTORY / "feature_label_contract.json"))
    manifest = {}
    frame_hashes = set()
    for split in ("train", "val"):
        for row in rows[split]:
            sid = row["sample_id"]
            meta = o0[sid]
            old = contract[sid]
            assert (split, row["prompt"], row["seed"]) == (old["split"], old["prompt"], old["video_seed"])
            root, source = p.paths(split, row)
            f0 = CACHE / "features" / f"{sid}.pt"
            t0 = ON / "o0" / sid / "T0.npz"
            t1 = t0.with_name("T1.npz")
            assert p.digest(source) == meta["source_sha256"]
            assert p.digest(f0) == old["feature_sha256"]
            assert p.digest(t0) == old["label_sha256"] == meta["arms"]["T0"]["label_sha256"]
            if split == "val":
                replay = OUT / "pre_a/A" / sid / "prediction.264"
                historic = HISTORY / "audit/A" / sid / "prediction.264"
                assert p.digest(replay) == p.digest(historic) == meta["arms"]["T1"]["source_reference_stream_sha256"]
                continue
            assert p.digest(t1) == meta["arms"]["T1"]["label_sha256"]
            base = p.p05.load_npz(root / "A/prediction.npz")
            mask = base["mode_id"] == 1
            np.testing.assert_array_equal(mask, p.p05.load_npz(CACHE / "labels" / f"{sid}.npz")["mode"] == 1)
            assert p.digest(root / "fixed_mode_mv.npz") == meta["arms"]["T1"]["mode_mv_sha256"]
            for path in (t0, t1):
                q = p.p05.load_npz(path)["q"]
                assert q.shape == (*p.SHAPE, 16) and not q[~mask].any()
                assert (np.count_nonzero(q, axis=(1, 2, 3, 4)) == 4).all()
                assert (np.count_nonzero(q, axis=-1) <= 1).all()
            hashes = digest_frames(source)
            assert len(set(hashes)) == 17 and not frame_hashes.intersection(hashes), sid
            frame_hashes.update(hashes)
            prompt_id = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            manifest[sid] = dict(video_id=sid, prompt_id=prompt_id, seed=row["seed"],
                source_hash=meta["source_sha256"], split="train_pool", frame_id=list(range(1, 17)),
                step_id=20, feature_layer="F0 frozen step20 / F1 cached final block / F2 clean latent",
                reference_trajectory_id={a: meta["arms"][a]["source_reference_stream_sha256"] for a in ("T0", "T1")},
                label_generator_version="963001586f01a4008307cf2221050f0de7b52e3d: O0/L1",
                qp=28, mode_mv_source="frozen A", candidate_mask_hash=hashlib.sha256(mask.tobytes()).hexdigest(),
                label_hash={"T0": p.digest(t0), "T1": p.digest(t1)},
                feature_hash={"F0": p.digest(f0), "F1": p.digest(row["hidden_path"]),
                              "F2": p.digest(row["final_latent_path"])},
                median_mv_px=float(np.median(np.linalg.norm(base["mv_qpel"][mask].astype(np.float32),axis=-1))/4),
                reason_for_grouping="exact manifest prompt identity, generated seed, distinct full-source and frame hashes")
    groups = {}
    for r in manifest.values():
        groups.setdefault(r["prompt_id"], []).append(r)
    assert len(groups) == 11 and all(len(g) == len({r["seed"] for r in g}) == 4 for g in groups.values())
    assert len({r["source_hash"] for r in manifest.values()}) == 44
    assert all(len({r["feature_hash"][a] for r in manifest.values()}) == 44 for a in ("F0","F1","F2"))
    results = []
    train = sample_rows(rows["train"], 8)
    data = {r["sample_id"]: data_for(ON/"o0", r, "T0") for r in train + rows["val"]}
    for name, width, params in (("c0-A",64,106784), ("c0-C",1504,1713824), ("c1-rank-C",1504,1713824)):
        for seed in CFG["seeds"]:
            folder = HISTORY/f"{name}-{seed}"
            snapshot = read(folder/"run.json")["snapshots"]["3200"]
            ck = folder/"head-3200.pt"
            assert p.digest(ck) == snapshot["head_sha256"]
            net = build(width).eval()
            net.load_state_dict(torch.load(ck, weights_only=True))
            assert sum(v.numel() for v in net.parameters()) == params
            scores = {}
            for split, clips in (("train",train), ("val",rows["val"])):
                scores[split] = [measure(net,data[r["sample_id"]],r["sample_id"]) for r in clips]
                assert [(x["sample"],x["hits"]) for x in scores[split]] == [(x["sample"],x["hits"]) for x in snapshot[split]]
            results.append(dict(run=folder.name, checkpoint_sha256=p.digest(ck), params=params, **scores))
            print(folder.name, *(mean(x["recall"] for x in scores[s]) for s in ("train","val")),flush=True)
    folds = []
    for f in CFG["folds"]:
        g = list(groups.values())
        chosen = g[f["prompt_offset"]:f["prompt_offset"] + CFG["support_prompts"]]
        tr = [sorted(group, key=lambda r:r["seed"])[f["seed_rank"]] for group in chosen]
        tids = [r["video_id"] for r in tr]
        near = lambda q: min(abs(q["median_mv_px"]-r["median_mv_px"]) for r in tr) <= 1
        split = dict(A=tids, B=[], C=[], D=[], D_far=[])
        for vid,r in manifest.items():
            if vid in tids:
                continue
            if r["prompt_id"] in {v["prompt_id"] for v in tr}:
                split["B"].append(vid)
            else:
                split["D"].append(vid)
                split["C" if near(r) else "D_far"].append(vid)
        assert not set(tids).intersection(split["B"] + split["D"])
        folds.append(split)
    p.save_json(OUT/"pre_history.json",dict(a=audit["baseline"], a_streams={
        r["sample"]:p.digest(OUT/"pre_a/A"/r["sample"]/"prediction.264") for r in audit["clips"]},
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), numpy=np.__version__,
        m1_units=units(), head_recompute=results, unique_source_frames=len(frame_hashes)))
    p.save_json(OUT/"split_manifest.json",dict(videos=list(manifest.values()),folds=folds))

if __name__ == "__main__":
    main()
