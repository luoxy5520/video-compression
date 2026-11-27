"""E2 original minimal adaptation; CompressAI 1.2.8 entropy APIs (BSD-3-Clause).
Not an implementation of complete GNVC-VD / GLVC / ELIC / DCVC.
"""
import hashlib, json, struct
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class Intra(nn.Module):
    def __init__(self, kind='I2', channels=32):
        super().__init__()
        if kind not in ('I2','I3'): raise ValueError(kind)
        self.kind, self.channels = kind, channels
        self.ga=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.GELU(),nn.Conv2d(32,channels,3,2,1))
        self.gs=nn.Sequential(nn.ConvTranspose2d(channels,32,4,2,1),nn.GELU(),nn.Conv2d(32,16,3,padding=1))
        self.eb=EntropyBottleneck(channels)
        if kind=='I3':
            self.ha=nn.Sequential(nn.Conv2d(channels,channels,3,2,1),nn.GELU(),nn.Conv2d(channels,channels,3,2,1))
            self.hs=nn.Sequential(nn.ConvTranspose2d(channels,channels,4,2,1),nn.GELU(),nn.ConvTranspose2d(channels,channels,4,2,1),nn.Softplus())
            self.gc=GaussianConditional(None)
    def forward(self,x):
        y=self.ga(x)
        if self.kind=='I2': yh,p=self.eb(y); probs=[p]
        else:
            zh,pz=self.eb(self.ha(y)); scales=self.hs(zh)[:,:,:y.shape[-2],:y.shape[-1]]+0.11
            yh,p=self.gc(y,scales); probs=[p,pz]
        return self.gs(yh)[:,:,:x.shape[-2],:x.shape[-1]], sum(-p.clamp_min(1e-9).log2().sum() for p in probs)/x.numel()
    def update(self):
        self.eb.update(force=True)
        if self.kind=='I3': self.gc.update_scale_table(torch.exp(torch.linspace(-2.2,5.5,64)),force=True)
    @torch.no_grad()
    def compress(self,x):
        y=self.ga(x)
        if self.kind=='I2': strings=[self.eb.compress(y)[0]]; shapes=[list(y.shape[-2:])]
        else:
            z=self.ha(y); zs=self.eb.compress(z); zh=self.eb.decompress(zs,z.shape[-2:]); scales=self.hs(zh)[:,:,:y.shape[-2],:y.shape[-1]]+0.11
            strings=[self.gc.compress(y,self.gc.build_indexes(scales))[0],zs[0]]; shapes=[list(y.shape[-2:]),list(z.shape[-2:])]
        return strings,shapes
    @torch.no_grad()
    def decompress(self,strings,shapes,hw):
        if self.kind=='I2': y=self.eb.decompress([strings[0]],shapes[0])
        else:
            z=self.eb.decompress([strings[1]],shapes[1]); scales=self.hs(z)[:,:,:shapes[0][0],:shapes[0][1]]+0.11
            y=self.gc.decompress([strings[0]],self.gc.build_indexes(scales))
        return self.gs(y)[:,:,:hw[0],:hw[1]]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_stream(path,model,x,model_id):
    strings,shapes=model.compress(x)
    h=dict(version=2,kind=model.kind,channels=model.channels,model_id=model_id,hw=list(x.shape[-2:]),shapes=shapes,lengths=list(map(len,strings)),rate_index=0,padding=[0,0],checksum=hashlib.sha256(b''.join(strings)).hexdigest())
    header=json.dumps(h,sort_keys=True,separators=(',',':')).encode()
    blob=b'LVC2'+struct.pack('<I',len(header))+header+b''.join(strings)
    Path(path).write_bytes(blob)
    return dict(bytes=len(blob),main_bytes=len(strings[0]),hyper_bytes=len(strings[1]) if len(strings)>1 else 0,metadata_bytes=8+len(header),sha256=sha(path))
def read_stream(path,model,model_id):
    b=Path(path).read_bytes()
    if len(b)<8 or b[:4]!=b'LVC2': raise ValueError('magic/truncated')
    n=struct.unpack('<I',b[4:8])[0]
    if n>65536 or n>len(b)-8: raise ValueError('header length')
    h=json.loads(b[8:8+n]); p=b[8+n:]
    if h['version']!=2 or h['model_id']!=model_id or h['kind']!=model.kind or h['channels']!=model.channels: raise ValueError('model/version mismatch')
    if any(type(v)!=int or v<=0 for v in h['lengths']) or sum(h['lengths'])!=len(p) or hashlib.sha256(p).hexdigest()!=h['checksum']: raise ValueError('payload/checksum')
    if len(h['hw'])!=2 or any(type(v)!=int or not 0<v<=4096 for v in h['hw']): raise ValueError('shape')
    strings=[]; off=0
    for size in h['lengths']: strings.append(p[off:off+size]); off+=size
    return model.decompress(strings,h['shapes'],h['hw'])
def load_model(path):
    c=torch.load(path,map_location='cpu',weights_only=True); m=Intra(c['kind'],c['channels'])
    for name, buf in m.named_buffers():
        if name in c['state']: buf.resize_(c['state'][name].shape)
    m.load_state_dict(c['state']); m.eval(); return m
if __name__=='__main__':
    import sys
    torch.set_num_threads(4)
    m=load_model(sys.argv[1]); torch.save(read_stream(sys.argv[2],m,sha(sys.argv[1])),sys.argv[3])
