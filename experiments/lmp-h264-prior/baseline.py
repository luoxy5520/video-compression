import hashlib, json, os, runpy, statistics, time
from pathlib import Path

HOME = Path.home()
KITS = [HOME/'senior-oracle-expand32-20261106', HOME/'a_new48_20261107/kit']
kit = KITS[1]
tools = runpy.run_path(str(kit/'linux/run_pipeline.py'))
run = tools['run']
sha = tools['sha']

def encode(sample, repeat=4):
    source_kit = KITS[int(sample.split('_')[1]) >= 1000]
    old = source_kit/'output/oracle_exports'/sample/'step_020'
    raw = old/'input_yuv420p.yuv'
    meta = json.loads((old/'metadata.json').read_text())
    assert sha(raw) == meta['input']['sha256']
    results = []
    for mode in (0, 3):
        folder = Path.cwd()/'baseline'/sample/str(mode)
        folder.mkdir(parents=True, exist_ok=True)
        bit, recon = folder/'stream.264', folder/'recon.yuv'
        args = [tools['ENC'], '-d', kit/'JM-backend/cfg/encoder_baseline.cfg',
                '-f', kit/'linux/skip_p16.cfg']
        params = dict(InputFile=raw, OutputFile=bit, ReconFile=recon,
                      SourceWidth=832, SourceHeight=480, FramesToBeEncoded=17,
                      FrameRate=16, QPISlice=28, QPPSlice=28, RDOptimization=0,
                      NumberReferenceFrames=1, PList0References=1, SearchMode=mode)
        for k, v in params.items():
            args += ['-p', f'{k} = {v}']
        env = {k:v for k,v in os.environ.items() if not k.startswith('JM_')}
        times, hashes = [], []
        for i in range(repeat):
            start = time.perf_counter()
            run(args, folder, folder/f'encode{i}.log', env)
            times.append(time.perf_counter()-start)
            hashes.append(sha(bit))
        assert len(set(hashes)) == 1
        jm, ff = tools['decode'](folder, bit, 'verified')
        assert sha(jm) == sha(ff) == sha(recon)
        result = dict(sample=sample, mode=mode, seconds=times,
                      median=statistics.median(times[1:]), sha256=hashes[0],
                      historical_match=sha(old/'reference.264')==hashes[0],
                      bytes=bit.stat().st_size)
        results.append(result)
        print(json.dumps(result), flush=True)
    return results

if __name__ == '__main__':
    import sys
    for sample in sys.argv[1:]:
        encode(sample)
