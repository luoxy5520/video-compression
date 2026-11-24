"""Generate evidence-linked report from actual saved rows; never synthesize measurements."""
import json,hashlib,datetime,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]; out=R/'reports'; out.mkdir(exist_ok=True)
def read(name):
 p=R/'results'/name; return json.loads(p.read_text()) if p.exists() else []
rows=read('rows.json'); sup=read('supplement.json'); val=read('validation_rows.json')
lines=['# Experiment 0 实验报告','', '## 1. Executive Summary','', '已实际实现并测试 Final Latent → LVC1 二进制码流 → 独立重建 Latent；编码模块不导入 VAE、H.264 或像素接口。原生 VAE 仅用于参考构建及接收端重建。以下结论受实际完成范围限制。','',f"Smoke 数据行 {len(rows)}；补充数据行 {len(sup)}；开发验证数据行 {len(val)}。",'', '状态以 results/*COMPLETE 文件和日志为准。', '', '| 样本 | 方法 | 完整码流 Bytes | PSNR-Y dB | 编码均值 ms |','|---|---|---:|---:|---:|']
for row in rows:
 if row['method'] in ('A0','A2','A3','B2','C2','D2','E1','E2_refresh4','x264_medium_18','x264_medium_28','x264_fast_28'):
  lines.append(f"| {row['sample']} | {row['method']} | {row['total_bytes']} | {row['psnr_y']:.3f} | {row['encode_s']['mean']*1000:.3f} |")
lines+=['','无损项的有限超大 PSNR 数值来自 MSE 下限数值保护，表示像素完全一致，不是实际有限信噪比。','', '## 2. Research Question','','验证真实码流、率失真、生成端及接收端负载分配，不要求优于 H.264。','','## 3. Motivation','','消除生成端像素物化与 VAE 解码，代价是接收端部署及运行原生 Wan VAE。','','## 4. Traditional Pipeline','','Final Latent → Wan VAE → RGB → BT709 limited YUV420p → JM/libx264 → H.264。','','## 5. Proposed Pipeline','','Final Latent → 对称量化 → 真正位打包/整数差分 → Zstd → LVC1。接收端独立解码后调用原始 VAE。','','## 6. Codec Implementation','','详见 codec_format.md、codec/lvc.py 和 tests/test_codec.py。量化参数、通道位宽及完整嵌套码流全部计入文件长度。未训练任何网络，未改变权重。','','## 7. Experiment Setup','','113Server，A100 80GB PCIe；复用 /home/luoxiangyu/Wan2.1/.venv。Wan 源码 9737cba9c1c3c4d04b33fcad41c111989865d315；模型 /mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth。原始样本配置与 SHA 见 reference.json。UniPC 20步、17帧、16fps、832×480。未重新生成 DiT 视频。','','两个开发样本：sample_0000 静态，sample_1004 局部运动。验证为 expand32 的4条val + new48的12条val；不接触test。模型权重不计入单视频码率，接收端须预先部署。模型首次部署、加载成本应另看 reference.json，网络传输未测。','','libx264 medium/fast，QP18/22/26/28/32/37/42，GOP17、B帧2、refs3；其他参数为对应preset默认值，日志记录实际命令。JM 原配置复用，编码汇总包含MP4封装，不能与不含封装的结果混为同口径。','','## 8. Latent Statistics','','各样本 statistics.json 包含 BCTHW shape、dtype、字节哈希、直方图、通道/时间均值方差与相关性。视频时间维 T 与去噪 timestep 无关。实际两条开发样本为 FP32；BF16 输入适配 NOT RUN。','','## 9. Rate-Distortion Results','','详见 results/rows.json、validation_rows.json、ssim.json 与 *_rd.png。展示 MP4 不参与正式码率或指标计算。PSNR基于原始RGB/YUV；SSIM仅在保存日志的配置上测量。LPIPS NOT RUN：环境无对应包与已核验权重。BD-Rate/BD-PSNR NOT RUN：尚未实现和审核有效重叠区间插值。']
for sample in sorted({r['sample'] for r in rows}):
 rr=[r for r in rows if r['sample']==sample]; d={r['method']:r for r in rr}
 if 'C2' in d and 'E1' in d:
  gain=100*(1-d['E1']['total_bytes']/d['C2']['total_bytes']); lines+=['',f"{sample}：E1 相对 C2 真实文件节省 {gain:.2f}%；相同量化符号及重建张量已 assert 完全相等。"]
 for m in ('B2','C2','D2'):
  if m not in d: continue
  h=sorted((r for r in rr if r['method'].startswith('x264_medium')),key=lambda r:r['psnr_y']); z=d[m]
  for a,b in zip(h,h[1:]):
   if a['psnr_y']<=z['psnr_y']<=b['psnr_y']:
    import math
    ratio=(z['psnr_y']-a['psnr_y'])/(b['psnr_y']-a['psnr_y']); estimate=math.exp(math.log(a['total_bytes'])+ratio*(math.log(b['total_bytes'])-math.log(a['total_bytes'])))
    lines+=['',f"{sample} {m}：在实测 H.264 PSNR 重叠区间作 log(rate)-PSNR 线性插值，码率约为 medium 的 {z['total_bytes']/estimate:.2f} 倍。此为插值分析，并非同画质直接实测或BD-Rate。"]
lines+=['','## 10. Latency Breakdown','','codec编解码6次（首轮预热，正式5次），报告Mean/Median/样本Std/Min/Max及原始计时。包含文件序列化和缓存文件I/O，不保证落盘fsync。参考的 vae_timing 字段测量 decode-to-CPU-RGB 完整调用；返回元组中的VAE秒数才是同步GPU VAE解码时间。模型加载时间单列。验证集H.264每配置仅一次，用于率失真，不作稳定性能结论。','','Final Latent→Bitstream 局部加速比：若benchmark.json存在则见文末本次直接测量表。Prompt→Bitstream 和 Prompt→Reconstruction：NOT RUN，历史流水线数字不拼接成本次端到端实测。接收端VAE成本没有消失。','','## 11. Memory Breakdown','','参考GPU allocated/reserved峰值见 reference.json；CPU峰值RSS和CPU/GPU量化D2H测试见 transfer.json。FP32→INT8使逻辑D2H字节数接近四分之一，但不是实际HBM/DRAM流量测量。编码端独立进程峰值内存对比 NOT RUN；完整系统峰值改善 NOT RUN；Nsight NOT RUN。其他GPU服务占用约8GB/卡，未停止它们。','','## 12. Temporal Redundancy Analysis','','E1使用int16差分存储后Zstd，可增加原始中间缓冲；收益依据完整文件。E2闭环预测引用重建上一切片，刷新2/4/8/All。每帧及末4帧指标见rows.json。17帧视频不足以验证长序列稳定性。F逐通道INT4扰动后，敏感度最强4通道保留INT8，其余INT4；此规则仅在开发样本选择，不宣称盲测。','','## 13. Failure Cases','','本机无numpy；复用服务器venv，不安装模型或依赖。首次本机py_compile因系统缓存路径写权限失败，后续服务器测试通过。一次工具脚本括号错误导致写入未执行，已更正。SSH在默认沙箱下被拒，随后使用已审核服务器连接执行。LPIPS不可用。所有未提供数据的指标均NOT RUN，不能以latent误差替代视频质量。','','## 14. Trade-off','','低位宽减少真实码流，但可能显著损失画质；时序差分是否有效依场景与真实文件长度判断。编码端消除VAE，并不等于完整系统消除VAE。','','## 15. Research Limitations','','未测实际网络；未测长视频；没有完全独占GPU。没有本次完整Prompt级流水线实测。GitHub推送 NOT RUN：当前实验目录未配置专用GitHub远端与大文件策略，不能推送到无关笔记仓库。','','## 16. Next Steps','','先补足同次实验端到端计时、LPIPS、完整内存剖析及更多运动场景，再判断是否研究联合优化；不得由Final Latent可量化推断可省略DiT去噪步骤。']
bench=read('benchmark.json')
if bench:
 import statistics
 lines+=['','## 本次直接测量的 Post-DiT 性能','', '| 样本 | 方法 | 编码均值 ms | Post-DiT编码加速比 | 含接收端均值 ms |','|---|---|---:|---:|---:|']
 for sample in sorted({r['sample'] for r in bench}):
  baseline=statistics.mean(r['post_dit_encoding_s'] for r in bench if r['sample']==sample and r['method']=='x264_medium_28')
  for method in ('x264_medium_28','INT8','INT4'):
   rr=[r for r in bench if r['sample']==sample and r['method']==method]
   enc=statistics.mean(r['post_dit_encoding_s'] for r in rr); full=statistics.mean(r['post_dit_to_receiver_pixels_s'] for r in rr)
   lines.append(f"| {sample} | {method} | {enc*1000:.3f} | {baseline/enc:.2f} | {full*1000:.3f} |")
 lines+=['','以上加速比较的量化方案与QP28并非质量匹配，不能作为同等画质性能胜出结论。传统接收端输出YUV，latent方案输出CPU RGB，显示准备口径亦不同。完整Prompt级两个加速比仍为NOT RUN。原始五次计时及阶段见benchmark.json；两端峰值未分进程隔离，不能据此声称峰值显存下降。']
if val:
 import statistics
 lines+=['','## 16条开发验证汇总','', '| 方法 | 已完成样本数 | 平均码流 Bytes | 平均PSNR-Y dB |','|---|---:|---:|---:|']
 for method in ('INT8','INT4','INT2','E1','E2','x264_medium_28','x264_fast_28'):
  rr=[r for r in val if r['method']==method]
  if rr: lines.append(f"| {method} | {len(rr)} | {statistics.mean(r['total_bytes'] for r in rr):.1f} | {statistics.mean(r['psnr_y'] for r in rr):.3f} |")
 lines+=['','此表为逐视频指标算术均值，非所有帧合并MSE后的PSNR。完整曲线见validation_rows.json，不以单点宣称新方案更优。']
(out/'final_report.md').write_text('\n'.join(lines)+'\n')
manifest=['# 文件清单','','| 文件路径 | 新增/复用 | 用途 |','|---|---|---|']
for p in sorted(R.rglob('*')):
 if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and 'tools_plot' not in p.parts:
  manifest.append(f'| {p.relative_to(R)} | 新增 | '+('实验代码' if p.suffix=='.py' else '配置/日志/报告/实测结果')+' |')
manifest+=['| /home/luoxiangyu/Wan2.1 | 复用 | 原生VAE及已有环境 |','| /mnt/data_hdd/luoxiangyu/dit_pairs | 复用 | 仅train/val Final Latent和参考YUV |','| /home/luoxiangyu/a_new48_20261107/kit | 复用 | 原始JM编码器与配置 |']
(out/'file_manifest.md').write_text('\n'.join(manifest)+'\n')
