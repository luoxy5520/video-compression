"""Diagnosis math must match the unchanged historical codec and symbol rules."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
class DiagnosticForward(unittest.TestCase):
    def test_training_forward_matches_default(self):
        torch.manual_seed(3);m=Intra('I3').train();x=torch.randn(2,16,60,104)
        torch.manual_seed(4);expected,r=m(x)
        torch.manual_seed(4);got=forward(m,x,'noise')
        self.assertTrue(torch.equal(expected,got['rec']));self.assertAlmostEqual(r.item(),got['rate'].item(),places=5)
    def test_hard_and_ste_forward_same_but_gradients_differ(self):
        torch.manual_seed(3);m=Intra('I3').eval();x=torch.randn(2,16,61,105)
        hard=forward(m,x,'hard');ste=forward(m,x,'ste')
        self.assertTrue(torch.equal(hard['rec'],ste['rec']));self.assertTrue(torch.equal(hard['yh'],ste['yh']))
        self.assertEqual(float((hard['yh']-hard['yh'].round()).abs().max()),0)
        m.zero_grad();ste['dist'].backward();self.assertGreater(sum(p.grad.norm().item() for p in m.ga.parameters()),0)

class ResumeAndTemporal(unittest.TestCase):
    def test_dynamic_cdf_restore(self):
        from diagnostic_codec import Codec
        source=Codec('R3');source.eb._quantized_cdf.resize_(32,40);source.eb._quantized_cdf.fill_(7)
        target=Codec('R3');restore_state(target,source.state_dict())
        self.assertTrue(torch.equal(target.eb._quantized_cdf,source.eb._quantized_cdf))
    def test_integral_mean_changes_cdf_not_canonical_symbols(self):
        import tempfile
        from temporal_entropy_model import Temporal
        for mode in ('TP1','TP2'):
            torch.manual_seed(7);m=Temporal(dict(variant='R3',step=.1,limit=255),mode).eval();m.base.update()
            with torch.no_grad():m.predictor[-1].bias[:32].fill_(1.6)
            x=torch.randn(3,16,12,16)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'stream';m.compress_video(x,path,'test-model');q,z=m.decompress_video(path,'test-model');fixed=m.fixed_sequence(x)
                self.assertTrue(torch.equal(q,torch.cat([v[0] for v in fixed])))
                self.assertTrue(torch.equal(z,torch.cat([m.base.gs(v[0]) for v in fixed])))

if __name__=='__main__':unittest.main()
