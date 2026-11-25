# Environment

113Server: NVIDIA A100 80GB PCIe ×2（实测 nvidia-smi）；复用 /home/luoxiangyu/Wan2.1/.venv。

```json
{
  "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.39",
  "torch": "2.7.1+cu128",
  "cuda": "12.8",
  "model_sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
}
```

模型 SHA256 由每次主扫描实际核验。已有其他GPU服务；质量任务可并发，不能直接将分片VAE耗时作为隔离性能结论。正式代表配置另行预热1次、测量5次。
本机临时venv仅用于轻量分析；本机缺libzstd，完整Codec测试在服务器执行。
