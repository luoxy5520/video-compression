# 代码变更索引

| ID | 文件 | 操作与功能 | 实验 | 验证 |
|---|---|---|---|---|
| LMP-01 | scripts/record.py | 新增命令分流、顺序、耗时和退出状态记录器 | 审计与构建 | 已写入本地与服务器日志 |
| LMP-02 | scripts/fixed_search.c.inc、scripts/build.py、configs/jm_fixed_search.patch | 新增固定整数预算。独立复制旧 JM，修改 EPZS_integer_motion_estimation，在冻结的 P16/List0/ref0 范围内，以真实重建参考计算 18 个唯一像素代价，再交回原生分数像素与闭环编码 | C-pixel-fixed、C1-fixed、C2-fixed | 编译、搜索计数断言、基线码流哈希、标准解码 |
| LMP-03 | configs/policy.json | 固定两条开发视频、QP28、32 px、候选模板和旧自适应映射 | C/D | 运行配置和哈希 |
| LMP-04 | scripts/smoke.py | 同一时钟测量先验生成和编码，输出每轮哈希、计数、双解码、PSNR 和码率 | A/C/D | 校验 C0 历史哈希和固定预算，重复码流一致 |
| LMP-05 | scripts/build.py | 从首次的 `CC='gcc -fcommon' OPT=3` 改为已知历史 `CFLAGS=-fcommon` | A/C | 首次重编失败证据与恢复后的 C0 哈希 |
| LMP-06 | scripts/smoke.py | 继续使用通过哈希验证的已有运行，将裸流解码和 MP4 帧率验证分开 | C/D | 保留初次探测失败，验证 16 帧/秒的 MP4 |
| LMP-07 | scripts/summarize.py | 将历史候选四格表与新结果联查，生成计时、RD 和门槛结果 | B/D/E | 结构化表与逐条重复记录可对照 |

| LMP-08 | scripts/profile_detail.py、configs/jm_phase_timing.patch | 在独立构建的 slice.c、macroblock.c、md_low.c、block.c 和 transform8x8.c 加探针，统计 P16、Skip 入口、变换量化、宏块写入与进程峰值 | A | 开启和关闭探针均匹配历史 C0 位流 |

归档提交见 `06_final_report.md` 的 Git 索引。源码以本次归档提交定位，旧基线源码来自审计记录所列的历史目录。
