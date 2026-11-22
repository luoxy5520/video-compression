"""Benchmark unmodified Wan generation followed by native JM B1 and MP4 remux."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WAN = Path.home() / 'Wan2.1'
CHECKPOINT = Path('/mnt/data_hdd/models/Wan2.1-T2V-1.3B')
GENERATE = dict(size=(832, 480), frame_num=17, shift=8, sample_solver='unipc',
                sampling_steps=20, guide_scale=6, offload_model=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def launch():
    # Child measures from immediately before spawning the fresh Python process.
    env = dict(os.environ, T2V_PROCESS_START=str(perf_counter()))
    return subprocess.call([sys.executable, __file__, *sys.argv[1:]], env=env)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('output', type=lambda p: Path(p).resolve())
    args.add_argument('sample', choices=['sample_0000', 'sample_1004'])
    args.add_argument('--repeats', type=int, default=5)
    args.add_argument('--profiles', type=int, default=5)
    options = args.parse_args()
    assert options.repeats >= 0 and options.profiles >= 0
    out = options.output
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    (out / 'runs').mkdir()
    process_start = float(os.environ['T2V_PROCESS_START'])
    logging.basicConfig(level=logging.INFO)
    import torch
    import numpy as np
    from profile_dit_steps import DiTSteps
    from profile_vae import VAEProfile
    from profile_encoder import CONFIGS, DEC, ENC, PARAMS, SIZE, encode, run, sha, verify

    sys.path[:0] = [str(WAN), str(WAN / 'experiments/baseline17')]
    from run_remote_baseline17 import sdpa
    from wan.modules import model
    from wan.text2video import FlowUniPCMultistepScheduler, WanT2V
    from wan.configs import WAN_CONFIGS

    model.flash_attention = sdpa
    torch.set_num_threads(4)
    sample = next(s for s in json.loads((ROOT / 'configs/samples.json').read_text())
                  if s['sample'] == options.sample)
    cfg = WAN_CONFIGS['t2v-1.3B']
    device = torch.device('cuda:0')

    def sync():
        torch.cuda.synchronize(device)

    load_start = perf_counter()
    torch.cuda.init()
    sync()
    cuda_init_s = perf_counter() - load_start
    load_start = perf_counter()
    pipe = WanT2V(cfg, str(CHECKPOINT), t5_cpu=False)
    sync()
    model_load_s = perf_counter() - load_start

    def request(run_id, kind):
        folder = out / 'runs' / run_id
        folder.mkdir()
        detailed = kind == 'profile'
        timings = {}
        boundaries = {}
        text_events = {}
        latent = []
        step = DiTSteps(torch, pipe.model, FlowUniPCMultistepScheduler)
        vae = VAEProfile(pipe.vae, step.event)

        def mark(name):
            sync()
            boundaries[name] = perf_counter()

        def wrap(original, before=None, after=None):
            def measured(*pos, **kw):
                if before:
                    before()
                result = original(*pos, **kw)
                if after:
                    after()
                return result
            return measured

        def text_call(original, *pos, **kw):
            result = original(*pos, **kw)
            fwd_start, fwd_end = text_events['text_forward_event_ms'][-1]
            text_events.setdefault('condition_build_event_ms', []).extend([
                (condition_start[-1], fwd_start), (fwd_end, step.event())])
            return result

        condition_start = []

        def text_part(original, name, gpu):
            def measured(*pos, **kw):
                begin = step.event() if gpu else perf_counter()
                result = original(*pos, **kw)
                end = step.event() if gpu else perf_counter()
                text_events.setdefault(name, []).append((begin, end))
                if name == 'tokenizer_cpu_ms':
                    condition_start.append(step.event())
                return result
            return measured

        original_decode = pipe.vae.decode

        def decode(z):
            if kind == 'cold_warmup':
                latent.append(z[0])
            if detailed:
                mark('vae_start')
            result = original_decode(z)
            if detailed:
                mark('vae_end')
            return result

        def tensor_to_rgb(video):
            if detailed:
                sync()
            begin = perf_counter()
            rgb_gpu = ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte()
            rgb_gpu = rgb_gpu.permute(1, 2, 3, 0).contiguous()
            if detailed:
                sync()
                timings['rgb_quantize_layout_s'] = perf_counter() - begin
            begin = perf_counter()
            rgb_cpu = rgb_gpu.cpu()
            if detailed:
                timings['rgb_d2h_s'] = perf_counter() - begin
            begin = perf_counter()
            rgb = rgb_cpu.numpy().tobytes()
            if detailed:
                timings['rgb_serialize_s'] = perf_counter() - begin
            return rgb

        steps = []
        torch.cuda.reset_peak_memory_stats(device)
        with ExitStack() as stack:
            if kind == 'cold_warmup' or detailed:
                stack.enter_context(patch.object(pipe.vae, 'decode', decode))
            if detailed:
                step.install()
                stack.callback(step.restore)
                vae.install()
                stack.callback(vae.restore)
                text = pipe.text_encoder
                stack.enter_context(patch.object(type(text), '__call__',
                    lambda instance, *pos, _call=type(text).__call__, **kw:
                        text_call(_call, instance, *pos, **kw)))
                stack.enter_context(patch.object(text.model, 'to',
                    wrap(text.model.to, before=lambda: mark('text_start'))))
                stack.enter_context(patch.object(text.model, 'cpu',
                    wrap(text.model.cpu, after=lambda: mark('text_end'))))
                stack.enter_context(patch.object(text, 'tokenizer',
                    text_part(text.tokenizer, 'tokenizer_cpu_ms', False)))
                stack.enter_context(patch.object(text.model, 'forward',
                    text_part(text.model.forward, 'text_forward_event_ms', True)))
                set_timesteps = FlowUniPCMultistepScheduler.set_timesteps

                def setup_scheduler(scheduler, *pos, **kw):
                    mark('dit_start')
                    t = perf_counter()
                    result = set_timesteps(scheduler, *pos, **kw)
                    timings['scheduler_setup_host_s'] = perf_counter() - t
                    return result

                stack.enter_context(patch.object(FlowUniPCMultistepScheduler, 'set_timesteps',
                                                  setup_scheduler))
                stack.enter_context(patch.object(pipe.model, 'cpu',
                    wrap(pipe.model.cpu, before=lambda: mark('dit_end'))))
            started_at = datetime.now().astimezone().isoformat()
            print(f'{sample["sample"]} {run_id} START {started_at}', flush=True)
            start = perf_counter()
            video = pipe.generate(sample['pairing']['prompt'], seed=sample['pairing']['seed'],
                                  **GENERATE)
            preprocess_start = perf_counter()
            timings['wan_generate_s'] = preprocess_start - start
            rgb = tensor_to_rgb(video)
            raw = folder / 'input.yuv'
            t = perf_counter()
            cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt',
                   'rgb24', '-s', '832x480', '-r', '16', '-i', 'pipe:0', '-vf',
                   'scale=in_range=full:out_range=limited:out_color_matrix=bt709',
                   '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
            yuv = run(cmd, folder, 'rgb_yuv', input=rgb, capture_stdout=True).stdout
            timings['rgb_yuv_pipe_s'] = perf_counter() - t
            assert len(yuv) == SIZE
            t = perf_counter()
            raw.write_bytes(yuv)
            timings['raw_write_s'] = perf_counter() - t
            timings['preprocess_s'] = perf_counter() - preprocess_start
            encode(folder, raw, timings)
            end = perf_counter()
            ended_at = datetime.now().astimezone().isoformat()
        timings['e2e_s'] = end - start
        if detailed:
            for stage in ('text', 'dit', 'vae'):
                timings[f'{stage}_s'] = boundaries[f'{stage}_end'] - boundaries[f'{stage}_start']
            timings['other_s'] = timings['e2e_s'] - sum(timings[f'{s}_s'] for s in
                        ('text', 'dit', 'vae', 'preprocess', 'h264', 'mux'))
            timings.update(vae.results())
            for key, entries in text_events.items():
                timings[key] = sum(a.elapsed_time(b) if 'event' in key else (b-a)*1000
                                   for a, b in entries)
            steps = step.results(sample['sample'], run_id)
            assert len(steps) == GENERATE['sampling_steps'] and len(timings['vae_chunk_event_ms']) == 5
        result = dict(sample_id=sample['sample'], run_id=run_id, kind=kind,
                      started_at=started_at, ended_at=ended_at,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                      **timings)
        if kind == 'cold_warmup':
            result.update(cold_e2e_s=end-process_start, model_load_s=model_load_s,
                          cuda_init_s=cuda_init_s)
        result.update(verify(folder, raw, sample))
        if latent:
            historical = Path(sample['latent'])
            assert sha(historical) == sample['latent_sha256']
            old = torch.load(historical, weights_only=True, map_location='cpu')
            new = latent[0].detach().cpu()
            result.update(historical_latent_equal=torch.equal(old, new),
                          historical_latent_max_abs=float((old-new).abs().max()))
            reference = np.fromfile(sample['final_yuv'], np.uint8).astype(np.int16)
            candidate = np.fromfile(raw, np.uint8).astype(np.int16)
            assert sha(sample['final_yuv']) == sample['input']['sha256']
            result['historical_raw_mismatch_bytes'] = int(np.count_nonzero(reference != candidate))
        write(folder / 'result.json', result)
        if steps:
            write(folder / 'steps.json', steps)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    pilot = request('cold_warmup', 'cold_warmup')
    # Decoder correctness is mandatory. A difference from the archived generation is
    # retained and reported, since reproducibility can differ across runtime builds.
    assets = [ENC, DEC, *CONFIGS, WAN / 'experiments/baseline17/run_remote_baseline17.py',
              CHECKPOINT / 'config.json']
    for p in CONFIGS:
        shutil.copyfile(p, out / p.name)
    write(out / 'environment.json', dict(
        sample=sample['sample'], generate=GENERATE, checkpoint=str(CHECKPOINT),
        model='Wan2.1-T2V-1.3B', negative_prompt=cfg.sample_neg_prompt,
        t5_cpu=False, t5='UMT5-XXL', t5_dtype=str(cfg.t5_dtype),
        dit_autocast=str(cfg.param_dtype), dit_weight_dtype=str(next(pipe.model.parameters()).dtype),
        vae_weight_dtype=str(next(pipe.vae.model.parameters()).dtype), vae_dtype=str(pipe.vae.dtype),
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
        python=sys.version, os=platform.platform(),
        gpu=torch.cuda.get_device_name(device), gpus_used=1,
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        wan_commit=subprocess.check_output(['git', '-C', WAN, 'rev-parse', 'HEAD'], text=True).strip(),
        gpu_inventory=subprocess.check_output(['nvidia-smi', '-L'], text=True).splitlines(),
        ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        cpu=subprocess.check_output(['lscpu'], text=True), jm_overrides=PARAMS,
        asset_sha256={str(p): sha(p) for p in assets}, argv=sys.argv))
    for kind, count in [('native', options.repeats), ('profile', options.profiles)]:
        for i in range(1, count + 1):
            result = request(f'{kind}_{i:02d}', kind)
            assert result['raw_sha256'] == pilot['raw_sha256']
            assert result['bit_sha256'] == pilot['bit_sha256']


if __name__ == '__main__':
    if 'T2V_PROCESS_START' not in os.environ:
        raise SystemExit(launch())
    main()
