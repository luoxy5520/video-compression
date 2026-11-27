"""Collect raw runs and produce the benchmark tables, figures and report."""
import argparse
import csv
import json
from pathlib import Path
import statistics

STAGES = [
    ('Text Encoder', 'text_s'), ('DiT Denoising', 'dit_s'), ('VAE Decoder', 'vae_s'),
    ('RGB/YUV Preprocess', 'preprocess_s'), ('H.264 Encoding', 'h264_s'),
    ('MP4 Muxing', 'mux_s'), ('Other / Overhead', 'other_s'), ('End-to-End', 'e2e_s'),
]
HISTORY = {
    'sample_0000': (1.0805, 1.0759, 1.1270),
    'sample_1004': (1.2128, 1.2193, 1.2696),
}
SAMPLES = tuple(HISTORY)


def read(path):
    return json.loads(path.read_text())


def csvout(path, rows, fields=()):
    with path.open('w') as f:
        keys = list(fields) or list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def percentile(values, q):
    v = sorted(values)
    x = (len(v) - 1) * q
    i = int(x)
    return v[i] + (v[min(i+1, len(v)-1)] - v[i]) * (x-i)


def describe(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                std=statistics.stdev(values) if len(values) > 1 else None,
                p95=percentile(values, .95), n=len(values))


def speed(original, encode, vae):
    return dict(encode_2x=original / (original-encode/2),
                encode_5x=original / (original-encode*.8),
                encode_unlimited=original / (original-encode),
                vae_and_encode_half=original / (original-(encode+vae)/2))


def figures(path, stages, steps, speeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    colors = ['#007F86', '#154979', '#DB7145', '#63A2C0', '#9367AC', '#E0AE46', '#9CABC0']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 135})
    fig, ax = plt.subplots(figsize=(9, 3.25), layout='constrained')
    left = [0] * len(SAMPLES)
    for (stage, _), color in zip(STAGES[:-1], colors):
        values = [next(r['mean'] for r in stages if r['sample_id'] == s and r['stage'] == stage)
                  for s in SAMPLES]
        ax.barh(SAMPLES, values, left=left, color=color, label=stage)
        left = [a+b for a, b in zip(left, values)]
    ax.invert_yaxis()
    ax.set(xlabel='Instrumented wall time (s)', title='Prompt to MP4 by stage')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.24), frameon=False)
    fig.savefig(path / 'pipeline_breakdown.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.9), layout='constrained')
    for s, color in zip(SAMPLES, colors):
        rows = [r for r in steps if r['sample_id'] == s]
        x = [r['step_index'] for r in rows]
        ax.plot(x, [r['mean'] for r in rows], 'o-', ms=3, color=color, label=f'{s} mean')
        ax.plot(x, [r['p95'] for r in rows], '--', color=color, alpha=.55, label=f'{s} P95')
    ax.set(xlabel='Denoising step', ylabel='CUDA event interval (ms)', title='UniPC step latency with CFG')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path / 'dit_step_latency.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')
    scenarios = [('encode_2x', 'H.264 2×'), ('encode_5x', 'H.264 5×'),
                 ('encode_unlimited', 'H.264 upper bound'), ('vae_and_encode_half', 'VAE + H.264 ½ time')]
    for i, s in enumerate(SAMPLES):
        row = next(r for r in speeds if r['sample_id'] == s)
        bars = ax.barh([j + (i-.5)*.32 for j in range(len(scenarios))],
                       [(row[k]-1)*100 for k, _ in scenarios], height=.30,
                       color=colors[i], label=s)
        ax.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=8)
    ax.set_yticks(range(len(scenarios)), [label for _, label in scenarios])
    ax.invert_yaxis()
    ax.set(xlabel='Theoretical throughput uplift (%)', title='Amdahl’s law applied to measured stage times')
    ax.margins(x=.22)
    ax.legend(frameon=False, loc='upper right')
    fig.savefig(path / 'theoretical_speedup.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    src, out = args.input, args.output
    dst = out / 'results'
    dst.mkdir(parents=True, exist_ok=True)
    chart_dir = out / 'figures'
    chart_dir.mkdir(exist_ok=True)
    raw, steps, environment = [], [], {}
    for sample in SAMPLES:
        env = src / sample / 'environment.json'
        if env.exists():
            environment[sample] = read(env)
        for p in sorted((src / sample / 'runs').glob('*/result.json')):
            raw.append(read(p))
            detail = p.with_name('steps.json')
            if detail.exists():
                steps.extend(read(detail))
        p = src / f'historical-{sample}' / 'result.json'
        if p.exists():
            raw.append(read(p))
    csvout(dst / 'raw_timings.csv', raw)
    csvout(dst / 'dit_step_timing.csv', steps, ('sample_id','run_id','step_index','timestep',
                                               'dit_forward_ms','scheduler_ms','total_step_ms'))
    (dst / 'environment.json').write_text(json.dumps(environment, indent=2, ensure_ascii=False) + '\n')
    stage_rows, step_rows, encoder, speeds = [], [], [], []
    for sample in SAMPLES:
        runs = [r for r in raw if r['sample_id'] == sample]
        for kind in ('native', 'profile'):
            group = [r for r in runs if r['kind'] == kind]
            if not group:
                continue
            end = statistics.mean(r['e2e_s'] for r in group)
            for label, key in STAGES:
                if key not in group[0]:
                    continue
                stats = describe([r[key] for r in group])
                stage_rows.append(dict(sample_id=sample, protocol=kind, stage=label, **stats,
                                       share_pct=stats['mean'] / end * 100))
            if kind == 'profile':
                avg = lambda key: statistics.mean(r[key] for r in group)
                speeds.append(dict(sample_id=sample, **speed(end, avg('h264_s'), avg('vae_s'))))
        for index in range(1, 21):
            group = [r for r in steps if r['sample_id'] == sample and r['step_index'] == index]
            if group:
                step_rows.append(dict(sample_id=sample, step_index=index, timestep=group[0]['timestep'],
                                      **describe([r['total_step_ms'] for r in group])))
        for kind in ('native', 'historical_raw_reencode'):
            group = [r for r in runs if r['kind'] == kind]
            if group:
                first = group[0]
                metrics = ('bitrate_kbps', 'bytes', 'p_y_psnr_db', 'all_y_psnr_db',
                           'historical_raw_match', 'historical_b1_match', 'decode_verified')
                encoder.append(dict(sample_id=sample, kind=kind, encoder='JM Native B1',
                                    encode_mean_s=statistics.mean(r['h264_s'] for r in group),
                                    n=len(group), **{k:first[k] for k in metrics}))
    csvout(dst / 'stage_summary.csv', stage_rows)
    csvout(dst / 'dit_step_summary.csv', step_rows)
    csvout(dst / 'encoder_summary.csv', encoder)
    csvout(dst / 'theoretical_speedup.csv', speeds)
    if len(speeds) == len(SAMPLES):
        figures(chart_dir, stage_rows, step_rows, speeds)
    report(out, src, raw, steps, environment, stage_rows, step_rows, encoder, speeds)


def report(out, src, raw, steps, environment, stages, step_rows, encoder, speeds):
    lines = []
    add = lines.append
    add('# Wan2.1 文生视频完整流水线性能基线')
    add('')
    add('## 实验环境与生成配置')
    add('')
    if environment:
        env = next(iter(environment.values()))
        add(f"113Server，{env['gpu']}，单卡，AMD EPYC 7302 双路，"
            f"Python {env['python'].split()[0]}，PyTorch {env['torch']}，"
            f"CUDA runtime {env['cuda_runtime']}，cuDNN {env['cudnn']}。"
            f"Wan 源码 Git 提交 `{env['wan_commit']}`。")
        add(f"UMT5-XXL {env['t5_dtype']}，推理时搬至 GPU，随后卸载到 CPU。"
            f"DiT 权重 {env['dit_weight_dtype']}，autocast {env['dit_autocast']}，"
            f"历史 masked PyTorch SDPA。VAE 权重 {env['vae_weight_dtype']}，"
            "输出 FP32。完整负向 prompt、参数和 SHA-256 见环境与配置文件。")
    add('Wan2.1-T2V-1.3B，832 × 480，17 帧，16 fps，UniPC 20 步，shift 8，CFG 6。'
        'CFG 在每步顺序执行 conditional 与 unconditional 两次 DiT 前向。'
        'VAE 按一个 latent 时间切片执行一次 decoder，共 5 片，复用因果缓存。')
    add('')
    add('| 样本 | Prompt | Seed |')
    add('| --- | --- | ---: |')
    for row in read(Path(__file__).resolve().parents[1] / 'configs/samples.json'):
        p = row['pairing']
        add(f"| {row['sample']} | {p['prompt']} | {p['seed']} |")
    add('')
    add('JM 19.0 FRExt 原生 B1，Baseline profile，QP I/P 28，EPZS SearchMode 3，'
        '搜索范围 32，1 个参考帧，首帧 IDR 后接 16 个 P 帧，CAVLC，RDOptimization 0。'
        '完整配置随实验归档。FFmpeg 做 BT.709 limited YUV420p 转换及 H.264 流复制封装。')
    add('')
    add('## 测量方法')
    add('')
    add('每个样本独立启动 Python，首个请求兼作 cold-start 和 warm-up。'
        'native 请求从调用 generate 前计时，到 FFmpeg 退出且 MP4 存在为止。'
        'profile 请求使用相同入口和参数，在 Text Encoder、DiT、VAE 边界同步该卡，'
        '另用 CUDA Event 记录同一流中的前向、调度和解码。逐步事件在整个请求完成后解析。'
        'CUDA Event 区间也可含流空闲和主机提交间隔。每次请求的解码校验和哈希均在计时之后。')
    add('Text Encoder 墙钟覆盖 T5 上下卡、正负 prompt tokenization、前向和条件构建。'
        'DiT 墙钟覆盖 scheduler 设置和去噪循环。DiT 下卡及清理、请求设置、latent 噪声初始化、'
        '编码器准备等落入 Other。VAE 墙钟从收到 final latent 开始，到浮点视频张量输出结束。'
        'RGB 量化、布局、传输、YUV 转换和文件写入计入预处理。'
        '流水线未执行 VAE Encoder 或独立的 DiT Encoder。')
    add('表 1 占比为同一组 profile 阶段均值除以其直接实测端到端均值，采用互斥墙钟。'
        'native 端到端另列。标准差为样本标准差，P95 用线性插值。')
    add('')
    add('## 表 1 各阶段耗时')
    for s in SAMPLES:
        add('')
        add(f'### {s}')
        add('')
        add('| Stage | Mean (s) | Median (s) | Std (s) | P95 (s) | 占端到端比例 | n |')
        add('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for r in stages:
            if r['sample_id'] != s:
                continue
            if r['protocol'] == 'native' and r['stage'] != 'End-to-End':
                continue
            label = r['stage'] + (' (native)' if r['protocol'] == 'native' else '')
            std = '—' if r['std'] is None else f"{r['std']:.4f}"
            add(f"| {label} | {r['mean']:.4f} | {r['median']:.4f} | {std} | "
                f"{r['p95']:.4f} | {r['share_pct']:.2f}% | {r['n']} |")
        add('')
        for r in raw:
            if r['sample_id'] == s and r['kind'] == 'cold_warmup':
                add(f"Cold-start {r['cold_e2e_s']:.3f} s，模型加载 {r['model_load_s']:.3f} s，"
                    f"显式 CUDA 初始化段 {r['cuda_init_s']:.3f} s。Wan 导入也可能初始化 CUDA，"
                    "包含于 cold-start 起始段。")
                add(f"首轮历史核验显示 final latent 逐元素相同为 {r['historical_latent_equal']}，"
                    f"最大绝对差 {r['historical_latent_max_abs']:.8g}，"
                    f"历史 YUV 有 {r['historical_raw_mismatch_bytes']} 字节不同。")
        profile = [r for r in raw if r['sample_id'] == s and r['kind'] == 'profile']
        if profile:
            avg = lambda key: statistics.mean(r[key] for r in profile)
            add(f"Tokenizer CPU {avg('tokenizer_cpu_ms'):.3f} ms，"
                f"UMT5 两次前向 CUDA Event {avg('text_forward_event_ms'):.3f} ms。"
                "文本条件构建与上下卡合在 Text Encoder 墙钟中。")
            add(f"VAE 核心 CUDA Event {avg('vae_core_event_ms'):.3f} ms，"
                f"5 片 decoder 合计 {statistics.mean(sum(r['vae_chunk_event_ms']) for r in profile):.3f} ms。"
                f"GPU RGB 量化及排布 {avg('rgb_quantize_layout_s')*1000:.3f} ms，"
                f"D2H {avg('rgb_d2h_s')*1000:.3f} ms，"
                f"CPU 字节序列化 {avg('rgb_serialize_s')*1000:.3f} ms，"
                f"FFmpeg 转换进程和管道 {avg('rgb_yuv_pipe_s')*1000:.3f} ms，"
                f"RAW write {avg('raw_write_s')*1000:.3f} ms，"
                f"编码准备 {avg('encoder_prepare_s')*1000:.3f} ms。"
                f"VAE 加预处理共 {avg('vae_s')+avg('preprocess_s'):.4f} s。")
            add(f"Formal profile PyTorch 峰值 allocated "
                f"{max(r['peak_allocated_mib'] for r in profile):.1f} MiB，"
                f"峰值 reserved {max(r['peak_reserved_mib'] for r in profile):.1f} MiB。")
    add('')
    add('## 表 2 DiT 逐步 CUDA Event 耗时')
    add('')
    for s in SAMPLES:
        group = [r for r in steps if r['sample_id'] == s]
        if not group:
            continue
        values = [r['total_step_ms'] for r in group]
        rows = [r for r in step_rows if r['sample_id'] == s]
        fast, slow = min(rows, key=lambda r:r['mean']), max(rows, key=lambda r:r['mean'])
        add(f"**{s}** 逐步总时间均值 {statistics.mean(values):.2f} ms，"
            f"最小/最大 {min(values):.2f}/{max(values):.2f} ms，"
            f"P50/P95 {statistics.median(values):.2f}/{percentile(values,.95):.2f} ms。"
            f"每步 scheduler 平均 {statistics.mean(r['scheduler_ms'] for r in group):.2f} ms。"
            f"最快/最慢的逐步均值出现在第 {fast['step_index']} / {slow['step_index']} 步，"
            f"均值差 {slow['mean']-fast['mean']:.2f} ms。")
        add('')
        add('| Step | Timestep | Mean (ms) | Std (ms) | P95 (ms) |')
        add('| ---: | ---: | ---: | ---: | ---: |')
        for r in rows:
            std = '—' if r['std'] is None else f"{r['std']:.2f}"
            add(f"| {r['step_index']} | {r['timestep']} | {r['mean']:.2f} | {std} | {r['p95']:.2f} |")
        add('')
    add('## 表 3 H.264 编码及解码校验')
    add('')
    add('| Sample | 输入 | Encoder | Encode Mean (s) | kb/s | Bytes | P 帧 Y-PSNR (dB) | 解码 |')
    add('| --- | --- | --- | ---: | ---: | ---: | ---: | --- |')
    for r in encoder:
        kind = '本次生成' if r['kind'] == 'native' else '历史 RAW，本次重编码'
        add(f"| {r['sample_id']} | {kind} | {r['encoder']} | {r['encode_mean_s']:.4f} | "
            f"{r['bitrate_kbps']:.2f} | {r['bytes']} | {r['p_y_psnr_db']:.4f} | "
            f"{'通过' if r['decode_verified'] else '失败'} |")
    add('')
    add('P 帧 Y-PSNR 以该次编码输入为参考，先汇总 16 个 P 帧的 8-bit Y MSE，再按峰值 255 换算。'
        '全 17 帧 Y-PSNR 另存 CSV。码率按 Annex B 字节数和 17/16 s 计算。'
        'JM 裸流解码、FFmpeg MP4 解码与 JM 重建逐字节一致，并验证了 MP4 帧数、帧率和分辨率。'
        '原生 JM 的初始化、ME、模式决策、变换量化熵编码和码流单独写入为 `not_available`。')
    add('')
    add('## 瓶颈与理论加速空间')
    add('')
    add('Amdahl 倍率为实测 profile 总时长除以按场景扣减目标阶段后的总时长。此处假设其余成本固定。')
    add('')
    add('| Sample | H.264 2× | H.264 5× | H.264 时间归零 | VAE 和 H.264 均减半 |')
    add('| --- | ---: | ---: | ---: | ---: |')
    for r in speeds:
        add(f"| {r['sample_id']} | {r['encode_2x']:.4f}× | {r['encode_5x']:.4f}× | "
            f"{r['encode_unlimited']:.4f}× | {r['vae_and_encode_half']:.4f}× |")
    for s in SAMPLES:
        group = sorted([r for r in stages if r['sample_id'] == s and r['protocol'] == 'profile'
                        and r['stage'] != 'End-to-End'], key=lambda r:r['mean'], reverse=True)
        if group:
            add('')
            add(f"{s} 的最大墙钟阶段为 {group[0]['stage']}（{group[0]['share_pct']:.2f}%），"
                f"其后为 {group[1]['stage']}（{group[1]['share_pct']:.2f}%）。")
    add('')
    add('## 历史对照')
    add('')
    add('| Sample | 历史 B1-measured (s) | 历史 DiT-gated 编码 (s) | 历史阶段中位数之和 (s) | 推算特征时间 / gated 编码 |')
    add('| --- | ---: | ---: | ---: | ---: |')
    for s, (native, gated, summed) in HISTORY.items():
        add(f'| {s} | {native:.4f} | {gated:.4f} | {summed:.4f} | {(summed-gated)/gated*100:.2f}% |')
    add('')
    add('历史特征开销由两列历史值相减，存在舍入误差。本次标准流水线未运行 DiT 辅助特征链。')
    add('')
    add('## 限制与复现')
    add('')
    add('1 秒采样的整卡 GPU 日志可识别持续负载，存在短任务漏采的可能。'
        '程序报告自身 PyTorch 峰值分配和保留显存。冷启动使用新进程，保留系统文件缓存。'
        'RAW 写入按 Python 写入返回计时，使用 OS 页缓存。RGB→YUV 计时含 FFmpeg 启停和管道复制。'
        'JM 编码时间含裸码流和重建写出，MP4 封装另计。')
    add('')
    add(f'服务器原始产物位于 `{src.resolve()}`。命令及更多计时字段见 [README](README.md)。')
    add('')
    text = '\n'.join(lines)
    (out / 'BASELINE_REPORT.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
