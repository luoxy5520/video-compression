# A1 三状态消融科研归档

[实验判定](decision.md) · [逐视频指标](per_video.csv) · [逐帧 PSNR](per_frame.csv) · [平均曲线](psnr.svg) · [运行清单](run_manifest.json) · [代码与 MP4](https://github.com/luoxy5520/video-compression/tree/codex/a1-three-state-20261109)

96 个评分码流，16 次 E 残差物化验证。原始逐组产物在 113Server 的 `~/a1_quality_20261108/A1_ThreeState/full`。`publish` 下的全量包按 16 条视频拆成 46 个 zstd 分卷，完整包含码流、编码器重建、两个标准解码输出、状态、日志、运行命令、原生 MC 和比特轨迹。输入原片、缓存 hidden、历史 checkpoint 由 `run_manifest.json` 给出服务器路径和 SHA-256。

从 113Server 上恢复全量包。

```bash
o="$HOME/a1_quality_20261108/A1_ThreeState/publish"
(cd "$o" && sha256sum -c PARTS.sha256)
mkdir -p a1-three-state-restored
cd a1-three-state-restored
for sample in sample_002{0,1,2,3} sample_102{4,5,6,7,8,9} sample_103{0,1,2,3,4,5}; do
  cat "$o/$sample.tar.zst.part-"* | zstd -dc | tar -xf -
done
sha256sum -c "$o/FILES.sha256"
```

`CODE_COMMIT` 记录交付代码，`run_manifest.json` 记录评分时的代码提交。`RAW_COMMIT` 记录全量分卷的提交对象，`FILES.sha256` 和 `PARTS.sha256` 分别校验原始文件与分卷。
