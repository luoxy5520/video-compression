"""Build GZ2 continuous targets on the existing frozen A reconstruction."""
from concurrent.futures import ThreadPoolExecutor
from common import *
from l1 import decode, labels
from sweep import luma

def clip(row):
    sid = row["sample_id"]
    root, source = p.paths("train",row)
    base = p.p05.load_npz(root/"A/prediction.npz")
    stream = root/"A/prediction.264"
    entry = manifest[sid]
    assert p.digest(stream) == entry["reference_trajectory_id"]["T1"]
    refs = decode(stream)
    source_y = p.p05.read_yuv(source)[0]
    target = labels(refs,source_y,base)
    sparse = p.p05.load_npz(ON/"o0"/sid/"T1.npz")["q"]
    np.testing.assert_array_equal(sparse,target["sparse"])
    residual = np.stack([source_y[t+1].astype(np.int16)-luma(ref,base["mv_qpel"][t]).astype(np.int16)
                         for t,ref in enumerate(refs)])
    y = residual.reshape(16,30,16,52,16).transpose(0,1,3,2,4)
    energy = np.square(y,dtype=np.float32).mean((-1,-2))
    path = folder/f"{sid}.npz"
    np.savez_compressed(path,energy=energy,low=target["continuous"],quantized=target["dense"][...,[0,1,4]])
    old = p.p05.load_npz(ON/"o0"/sid/"T0.npz")["q"]
    common = (old!=0)&(sparse!=0)
    count = int(common.sum())
    return dict(video_id=sid, label_sha256=p.digest(path), reference=entry["reference_trajectory_id"]["T1"],
                t0_t1_overlap=count, t0_t1_iou=count/(128-count),
                t0_t1_sign=float((np.sign(old[common])==np.sign(sparse[common])).mean()) if count else None,
                t0_t1_level_mae=float(np.abs(old[common]-sparse[common]).mean()) if count else None)

if __name__ == "__main__":
    torch.set_num_threads(1)
    manifest = {x["video_id"]:x for x in read(OUT/"split_manifest.json")["videos"]}
    folder = OUT/"continuous"
    folder.mkdir()
    with ThreadPoolExecutor(max_workers=5) as pool:
        metrics = list(pool.map(clip,p.rows_by_split()["train"]))
    p.save_json(OUT/"gz2_labels.json",metrics)
