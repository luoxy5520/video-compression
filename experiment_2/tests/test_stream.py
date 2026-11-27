"""Round-trip and negative checks for prototype stream / checkpoint contract."""
import sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codec import Intra,write_stream,read_stream,load_model,sha
class StreamTests(unittest.TestCase):
    def test_real_stream_and_checkpoint_odd_shapes(self):
        torch.set_num_threads(4)
        for kind in ('I2','I3'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                d=Path(d);m=Intra(kind).eval();m.update();ck=d/'model.pt';torch.save(dict(kind=kind,channels=32,state=m.state_dict()),ck)
                m=load_model(ck);x=torch.randn(1,16,61,105);p=d/'stream';stats=write_stream(p,m,x,sha(ck));out=read_stream(p,m,sha(ck))
                with torch.no_grad():expected,_=m(x)
                self.assertEqual(out.shape,x.shape);self.assertTrue(torch.equal(out,expected));self.assertEqual(stats['bytes'],p.stat().st_size)
                with self.assertRaises(ValueError):read_stream(p,m,'wrong checkpoint')
                original=p.read_bytes()
                for bad in (b'',original[:7],original[:-1],original[:-1]+bytes([original[-1]^1])):
                    p.write_bytes(bad)
                    with self.assertRaises(ValueError):read_stream(p,m,sha(ck))
if __name__=='__main__':unittest.main()
