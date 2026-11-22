"""Separate RGB datatype conversion, layout, and FFmpeg CPU time."""

def profile_rgb_layout():
    """Independently split RGB preparation on verified cached final latents."""
    import argparse
    import hashlib
    import json
    import os
    import resource
    from pathlib import Path
    import sys
    from time import perf_counter

    import torch

    from profile_encoder import run, sha

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    inputs = json.loads((root / 'configs/samples.json').read_text())
    sys.path.insert(0, str(Path.home() / 'Wan2.1'))
    from wan.modules.vae import WanVAE

    torch.set_num_threads(4)
    vae = WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                 device='cuda:0')
    all_rows = []
    for sample in inputs:
        folder = args.output / sample['sample']
        folder.mkdir()
        assert sha(sample['latent']) == sample['latent_sha256']
        latent = torch.load(sample['latent'], weights_only=True, map_location='cuda:0')
        with torch.no_grad():
            video = vae.decode([latent])[0]
        assert video.dtype == torch.float32 and video.shape == (3, 17, 480, 832)
        for i in range(6):
            start, quantized, arranged = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda.synchronize(0)
            start.record()
            uint8 = ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte()
            quantized.record()
            thwc = uint8.permute(1, 2, 3, 0).contiguous()
            arranged.record()
            arranged.synchronize()
            t = perf_counter()
            rgb = thwc.cpu().numpy().tobytes()
            transfer_and_pack_s = perf_counter() - t
            cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt',
                   'rgb24', '-s', '832x480', '-r', '16', '-i', 'pipe:0',
                   '-vf', 'scale=in_range=full:out_range=limited:out_color_matrix=bt709',
                   '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
            cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            conversion_start = perf_counter()
            result = run(cmd, folder, f'convert_{i}', input=rgb, capture_stdout=True)
            conversion_wall_s = perf_counter() - conversion_start
            cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            assert hashlib.sha256(result.stdout).hexdigest() == sample['input']['sha256']
            all_rows.append(dict(sample_id=sample['sample'], repeat=i, warmup=i==0,
                                 quantize_event_ms=start.elapsed_time(quantized),
                                 layout_event_ms=quantized.elapsed_time(arranged),
                                 transfer_and_pack_s=transfer_and_pack_s,
                                 conversion_wall_s=conversion_wall_s,
                                 ffmpeg_user_cpu_s=cpu_after.ru_utime-cpu_before.ru_utime,
                                 ffmpeg_system_cpu_s=cpu_after.ru_stime-cpu_before.ru_stime,
                                 historical_raw_match=True,
                                 cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES')))
            del uint8, thwc, rgb
        del video, latent
    (args.output / 'rgb_detail.json').write_text(json.dumps(all_rows, indent=2) + '\n')
    print(json.dumps(all_rows), flush=True)


if __name__ == '__main__':
    profile_rgb_layout()
