"""Audit and finalize mandatory metadata; remeasure codec when changing header."""
from pathlib import Path
source=Path(__file__).with_name('run_smoke.py').read_text(); exec(source[:source.index('for bits in (2,3,4,5,6,8):')])
assert (OUT/'VALIDATION_COMPLETE').exists()
for jsonname in ('rows.json','validation_rows.json','supplement.json'):
    path=OUT/jsonname; rows=json.loads(path.read_text())
    for row in rows:
        method=row['method']; sample=row['sample']; folder=OUT/sample
        if method.startswith('JM') or method.startswith('x264'): continue
        filename='mixed' if method=='F_mixed' else method; stream=folder/(filename+'.lvc')
        if not stream.exists(): continue
        import struct
        blob=stream.read_bytes(); n=struct.unpack('<I',blob[4:8])[0]; h=json.loads(blob[8:8+n]); old=decode(stream)
        ref=json.loads((folder/'reference.json').read_text()); meta=ref.get('source',ref.get('metadata')); source_latent=meta.get('final_latent',meta.get('latent'))
        z=torch.load(source_latent,weights_only=True,map_location='cpu').numpy()[None]
        meta=dict(meta,vae='Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',vae_sha256='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',normalization='solver final latent unchanged; native WanVAE.decode applies self.scale internally',width=832,height=480,frames=17,fps=16)
        cfg=dict(channel_bits=h['channel_bits']) if h.get('mixed') else h['quantization']; compression=dict(algorithm='zstd') if h.get('mixed') else h['compression']
        sizes,et=timing(lambda:encode(z,stream,cfg,compression,meta)); new,dt=timing(lambda:decode(stream)); assert np.array_equal(new,old)
        row.update(**sizes,encode_s=et,decode_s=dt,bitrate_kbps=sizes['total_bytes']*8/(17/16)/1000,bpp=sizes['total_bytes']*8/(17*480*832),compression_ratio=z.nbytes/sizes['total_bytes'],metadata_finalized=True)
    path.write_text(json.dumps(rows,indent=2))
(OUT/'METADATA_COMPLETE').touch()
