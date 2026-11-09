# GitHub 科研指标归档

包含报告、配置、分组清单、逐视频派生指标和服务器完整包的校验清单。完整包的三个分卷、H.264 码流、JM 原始输出、逐候选样本和 checkpoint 均在 113Server。

在空目录下将派生指标解卷。随后用 DERIVED.sha256 核验。

```sh
sha256sum -c FILES.sha256
sha256sum -c PARTS.sha256
cat metrics.tar.zst.part-* | zstd -dc | tar -xf -
sha256sum -c DERIVED.sha256
```

CORE.sha256、STUDENTS.sha256 和 SERVER_PARTS.sha256 适用于服务器完整包。GitHub 的指标分卷使用 PARTS.sha256。
