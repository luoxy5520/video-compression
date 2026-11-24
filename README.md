# E0 Direct Latent Compression

实际实验产物位于113Server `/home/luoxiangyu/E0_Direct_Latent_Compression`；本目录保存源码、日志、报告、指标和代表性展示。原始latent/YUV、全量码流和中间产物保留服务器，避免拉取大文件。

先阅读 reports/final_report.md，运行方式与码流规范见 reports/codec_format.md。不能将未执行的Prompt级端到端性能当作局部codec测试结果。

原模型权重不修改，不训练，不接触test数据。正式PSNR在展示MP4再编码前计算。
