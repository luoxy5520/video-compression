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
if __name__=='__main__':unittest.main()
