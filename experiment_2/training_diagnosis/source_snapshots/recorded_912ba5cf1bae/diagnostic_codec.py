"""Small gated adaptation of existing I3: same hyperprior and native rANS.
Optional no-downsample transforms are the E2.1b R3 experiment, not a new large CNN.
"""
from common import *
from torch import nn
class ScaleBound(nn.Module):
    def __init__(self,step=.1,limit=255,inverse=False):super().__init__();self.step=step;self.limit=limit;self.inverse=inverse
    def forward(self,x):return x*self.step if self.inverse else (x/self.step).clamp(-self.limit,self.limit)
class Codec(Intra):
    def __init__(self,variant='R1',step=.1,limit=255):
        super().__init__('I3',64 if variant=='R2' else 32);self.config=dict(variant=variant,step=step,limit=limit)
        if variant.startswith('R3'):
            self.ga=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1))
            self.gs=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,16,3,padding=1))
        self.ga=nn.Sequential(self.ga,ScaleBound(step,limit));self.gs=nn.Sequential(ScaleBound(step,limit,True),self.gs)
    def from_reconstruction(self,path):
        c=torch.load(path,weights_only=False,map_location='cpu');state=c['state']
        self.ga[0].load_state_dict({k[3:]:v for k,v in state.items() if k.startswith('ga.')});self.gs[1].load_state_dict({k[3:]:v for k,v in state.items() if k.startswith('gs.')})
def load(path):
    c=torch.load(path,weights_only=False,map_location='cpu');m=Codec(**c['config'])
    for name,buf in m.named_buffers():
        if name in c['state']:buf.resize_(c['state'][name].shape)
    m.load_state_dict(c['state']);return m.eval(),c
if __name__=='__main__':
    m,c=load(sys.argv[1]);torch.save(read_stream(sys.argv[2],m,sha(sys.argv[1])),sys.argv[3])
