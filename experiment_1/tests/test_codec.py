import unittest,tempfile,sys,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from codec.lvc import encode,decode,pack,unpack
class CodecTests(unittest.TestCase):
    def test_packing(self):
        for bits in range(2,9):
            q=np.arange(-(1<<(bits-1)),1<<(bits-1),dtype='int16')
            for n in (1,3,len(q)-1,len(q)):
                self.assertTrue(np.array_equal(q[:n],unpack(pack(q[:n],bits),bits,n)))
    def test_strict_packing(self):
        for bits in range(2,9):
            for n in (0,1,3,7,9,101):
                q=np.zeros(n,dtype='int16'); data=pack(q,bits)
                self.assertEqual(len(data),(n*bits+7)//8)
                self.assertTrue(np.array_equal(unpack(data,bits,n),q))
                with self.assertRaises(ValueError): unpack(data+b'\x00',bits,n)
                if n*bits%8:
                    bad=bytearray(data); bad[-1]|=128
                    with self.assertRaises(ValueError): unpack(bad,bits,n)
            for bad in (-32769,32768,65536):
                with self.assertRaises(ValueError): pack(np.array([bad]),bits)
        for bits in (1,9,2.5,True):
            with self.assertRaises(ValueError): pack(np.array([0]),bits)

    def test_profile_preserves_bitstream(self):
        z=np.random.default_rng(12).normal(size=(1,3,2,3,5)).astype('float32')
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a'; b=Path(d)/'b'
            for cfg in [dict(bits=i,granularity='channel') for i in range(2,9)]+[dict(channel_bits=[3,5,7])]:
                encode(z,a,cfg,dict(algorithm='zstd'),{})
                profile={};encode(z,b,cfg,dict(algorithm='zstd'),{},profile)
                self.assertEqual(a.read_bytes(),b.read_bytes())
                self.assertTrue(all(profile[k]>=0 for k in ('quantization_s','packing_s','entropy_s','total_s')))
                self.assertGreaterEqual(profile['total_s'],sum(profile[k] for k in ('quantization_s','packing_s','entropy_s')))

    def test_roundtrip(self):
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.lvc'
            for dtype in ('float16','float32','float64'):
                z=rng.normal(size=(1,3,5,4,7)).astype(dtype)
                for alg in ('none','zlib','zstd'):
                    for shuffle in (False,True):
                        encode(z,p,{},dict(algorithm=alg,shuffle=shuffle),{})
                        self.assertEqual(z.tobytes(),decode(p).tobytes())
            z=z.astype('float32')
            for bits in range(2,9):
                cfg=dict(bits=bits,granularity='channel'); encode(z,p,cfg,dict(algorithm='zstd'),{}); direct=decode(p)
                cfg['temporal']='integer_delta'; encode(z,p,cfg,dict(algorithm='zstd'),{})
                self.assertTrue(np.array_equal(direct,decode(p)))
                for refresh in (0,2,4,8):
                    cfg.update(temporal='closed_loop',refresh=refresh); encode(z,p,cfg,dict(algorithm='zstd'),{}); self.assertTrue(np.isfinite(decode(p)).all())
            encode(z,p,dict(channel_bits=[8,4,2]),dict(algorithm='zstd'),{})
            self.assertEqual(decode(p).shape,z.shape)
            # Fresh process has only the bitstream, codec code and numpy.
            subprocess.run([sys.executable,'-c','from codec.lvc import decode; import sys; print(decode(sys.argv[1]).shape)',str(p)],check=True,cwd=Path(__file__).resolve().parents[1])
            data=bytearray(p.read_bytes()); data[-1]^=1; p.write_bytes(data)
            with self.assertRaises(ValueError): decode(p)
if __name__=='__main__': unittest.main()
