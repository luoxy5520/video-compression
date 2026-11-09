#!/usr/bin/env bash
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
code=$(git rev-parse HEAD)
r=$HOME/a1_quality_20261108/P2_GeneralizationProbe
d=experiments/a1_quality_20261108/P2_GeneralizationProbe
a=$r/publish
mkdir "$a"
printf '%s\n' "$code" > "$a/CODE_COMMIT"
git -C "$repo" archive "$code" -- "$d" | tar -xf - -C "$a"
cd "$r"
{
    printf '%s\n' pre_history.json split_manifest.json gz1.json gz1_native.json gz2.json gz2_labels.json historical_archive_refs.log
    find gz0-*-*-* -name metrics.json -o -name head.pt
    find continuous gz1_pairs gz1_native_pairs gz2_pairs -type f
    find pre_a/A -type f \( -name prediction.264 -o -name bits.csv -o -name coeff.bin -o -name adapter.json -o -name encode.log -o -name jm_decode.log \)
    printf '%s\n' pre_a/audit.json
} | LC_ALL=C sort > "$a/CORE.files"
xargs -d '\n' sha256sum < "$a/CORE.files" > "$a/CORE.sha256"
find gz0-*-*-* -name head.pt | LC_ALL=C sort | xargs -d '\n' sha256sum > "$a/STUDENTS.sha256"
sha256sum -c "$a/CORE.sha256" >/dev/null
tar -cf - -T "$a/CORE.files" | zstd -T4 -9 | split -b 40M -d -a 3 - "$a/core.tar.zst.part-"
cd "$a"
sha256sum core.tar.zst.part-* > PARTS.sha256
sha256sum -c PARTS.sha256
cat core.tar.zst.part-* | zstd -dc | tar -tf - | LC_ALL=C sort > PARTS.contents
cmp CORE.files PARTS.contents
cat > SERVER_ONLY.tsv <<'EOF'
path_scope	reason
~/a1_quality_20261108/P2_GeneralizationProbe/pre_a/A/*/{*.yuv,trace.bin,replay.bin,prediction.npz,i_and_neutral_p.bin}	Raw reconstructions and regenerable JM state
~/a1_quality_20261108/P2_GeneralizationProbe/*.log	Full server run logs
~/a1_quality_20261108/P2_InfoProbe/g_labels	Existing historical derived data
~/a1_quality_20261108/P2_OnPolicy/o0	Existing historical label trajectories and label caches
~/a1_quality_20261108/P2_CapacityProbe	Existing historical checkpoints and replay cache
dataset manifests: source_export_dir, hidden_path, final_latent_path	Source YUV, DiT hidden and clean latent remain on 113Server
/mnt/data_hdd/models/dit_compression	Existing model weights
EOF
cat > README.md <<'EOF'
# P2-GeneralizationProbe 科研归档

归档包含 8 个诊断头 checkpoint、逐视频和候选对指标、连续目标、冻结 A 的 16 条 H.264 回放码流及 JM 分析文件。源码提交见 CODE_COMMIT。SERVER_ONLY.tsv 列出留在 113Server 的输入和可再生输出。

从本目录校验及解卷：

```sh
sha256sum -c FILES.sha256
sha256sum -c PARTS.sha256
mkdir core
cat core.tar.zst.part-* | zstd -dc | tar -xf - -C core
(cd core && sha256sum -c ../CORE.sha256 && sha256sum -c ../STUDENTS.sha256)
```
EOF
find . -type f ! -name FILES.sha256 | LC_ALL=C sort | xargs -d '\n' sha256sum > FILES.sha256
sha256sum -c FILES.sha256 >/dev/null
