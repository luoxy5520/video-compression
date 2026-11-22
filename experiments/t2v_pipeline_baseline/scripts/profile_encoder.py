"""Run the historical B1 encoder and verify the generated reconstruction."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

KIT = Path.home() / 'a_new48_20261107/kit'
ENC = KIT / 'linux/JM/bin/lencod.exe'
DEC = KIT / 'linux/JM/bin/ldecod.exe'
CONFIGS = [KIT / 'JM-backend/cfg/encoder_baseline.cfg', KIT / 'linux/skip_p16.cfg',
           KIT / 'JM-backend/cfg/decoder.cfg']
SIZE = 832 * 480 * 17 * 3 // 2
PARAMS = dict(SourceWidth=832, SourceHeight=480, FramesToBeEncoded=17,
              FrameRate=16, QPISlice=28, QPPSlice=28, RDOptimization=0,
              NumberReferenceFrames=1, PList0References=1, SearchMode=3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, folder, name, capture_stdout=False, **kwargs):
    args = list(map(str, args))
    with (folder / 'commands.jsonl').open('a') as log:
        log.write(json.dumps(args) + '\n')
    with (folder / f'{name}.log').open('w') as log:
        return subprocess.run(args, cwd=folder, stderr=log, stdout=subprocess.PIPE if capture_stdout else log, check=True, **kwargs)


def encode(folder, raw, timings):
    t = perf_counter()
    assert raw.stat().st_size == SIZE
    bit, recon = folder / 'stream.264', folder / 'recon.yuv'
    params = dict(InputFile=raw, OutputFile=bit, ReconFile=recon, **PARAMS)
    args = [ENC, '-d', CONFIGS[0], '-f', CONFIGS[1]]
    for key, val in params.items():
        args.extend(['-p', f'{key} = {val}'])
    env = {k: v for k, v in os.environ.items() if not k.startswith('JM_')}
    timings['encoder_prepare_s'] = perf_counter() - t
    t = perf_counter()
    run(args, folder, 'encode', env=env)
    timings['h264_s'] = perf_counter() - t
    t = perf_counter()
    mp4 = folder / 'video.mp4'
    run(['ffmpeg', '-nostdin', '-v', 'error', '-fflags', '+genpts', '-framerate', '16', '-i', bit,
         '-map', '0:v:0', '-c:v', 'copy', mp4], folder, 'mux')
    timings['mux_s'] = perf_counter() - t
    assert mp4.stat().st_size > 0


def verify(folder, raw, sample):
    bit, recon = folder / 'stream.264', folder / 'recon.yuv'
    jm, ff = folder / 'verify_jm.yuv', folder / 'verify_mp4.yuv'
    run([DEC, '-d', CONFIGS[2], '-p', f'InputFile = {bit}', '-p',
         f'OutputFile = {jm}', '-p', 'RefFile = ""'], folder, 'verify_jm')
    run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', folder / 'video.mp4',
         '-pix_fmt', 'yuv420p', '-f', 'rawvideo', ff], folder, 'verify_mp4')
    assert jm.stat().st_size == ff.stat().st_size == recon.stat().st_size == SIZE
    assert sha(jm) == sha(ff) == sha(recon)
    probe_args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                  'stream=codec_name,width,height,avg_frame_rate,nb_frames',
                  '-of', 'json', str(folder / 'video.mp4')]
    run(probe_args, folder, 'ffprobe')
    stream, = json.loads((folder / 'ffprobe.log').read_text())['streams']
    assert (stream['codec_name'], stream['width'], stream['height'],
            stream['avg_frame_rate'], stream['nb_frames']) == ('h264', 832, 480, '16/1', '17')
    pixels = 832 * 480
    a, b = (np.fromfile(p, np.uint8).reshape(17, -1)[:, :pixels].astype(np.float64)
            for p in (raw, recon))
    mse = ((a - b)**2).mean(axis=1)
    digest, source = sha(bit), sha(raw)
    old = json.loads(Path(sample['historical_b1_result']).read_text())
    return dict(bytes=bit.stat().st_size, mp4_bytes=(folder / 'video.mp4').stat().st_size,
                bitrate_kbps=bit.stat().st_size * 8 * 16 / 17 / 1000,
                all_y_psnr_db=float(10 * np.log10(255**2 / mse.mean())),
                p_y_psnr_db=float(10 * np.log10(255**2 / mse[1:].mean())),
                bit_sha256=digest, raw_sha256=source, recon_sha256=sha(recon),
                historical_raw_match=source == sample['input']['sha256'],
                historical_b1_match=digest == old['sha256'], decode_verified=True,
                encoder_init_s='not_available', me_s='not_available',
                mode_decision_s='not_available', transform_entropy_s='not_available',
                bitstream_write_s='not_available')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=lambda p: Path(p).resolve())
    parser.add_argument('sample', choices=['sample_0000', 'sample_1004'])
    args = parser.parse_args()
    samples = json.loads((Path(__file__).resolve().parents[1] / 'configs/samples.json').read_text())
    sample = next(row for row in samples if row['sample'] == args.sample)
    raw = Path(sample['raw'])
    assert sha(raw) == sample['input']['sha256']
    args.output.mkdir(parents=True)
    timings = {}
    encode(args.output, raw, timings)
    result = dict(sample_id=args.sample, kind='historical_raw_reencode',
                  **timings, **verify(args.output, raw, sample))
    assert result['historical_b1_match']
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
