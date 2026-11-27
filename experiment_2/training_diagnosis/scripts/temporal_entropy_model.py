"""Fixed canonical integer symbols with learned temporal CDF translation.
Integer predicted means shift CDF origins only: round(q-k)+k=q for integer q,k.
No transformation or dequantization changes; all references are reconstructed.
"""
from common import *
from diagnostic_codec import Codec
from torch import nn
import struct
class Temporal(nn.Module):
    def __init__(self,config,mode='TP1'):
        super().__init__();self.config=config;self.mode=mode;self.base=Codec(**config).eval().requires_grad_(False)
        if mode!='TP0':
            n=self.base.channels;self.predictor=nn.Sequential(nn.Conv2d(n+(16 if mode=='TP1' else n),32,3,padding=1),nn.GELU(),nn.Conv2d(32,2*n,3,padding=1));nn.init.zeros_(self.predictor[-1].weight);nn.init.zeros_(self.predictor[-1].bias)
    def params_for(self,hs,previous,training=False):
        if previous is None or self.mode=='TP0':return torch.zeros_like(hs),hs
        raw=self.predictor(torch.cat([hs,previous],1));mean,logscale=raw.chunk(2,1);mean=mean.clamp(-255,255);integer=mean.round();mean=mean+(integer-mean).detach() if training else integer
        return mean,(hs*logscale.clamp(-3,3).exp()).clamp_min(.11)
    def previous(self,q):return self.base.gs(q) if self.mode=='TP1' else q
    @torch.no_grad()
    def fixed_sequence(self,x):
        frames=[]
        for frame in x:
            y=self.base.ga(frame[None]);z=self.base.ha(y);zs=self.base.eb.compress(z);zh=self.base.eb.decompress(zs,z.shape[-2:]);hs=self.base.hs(zh)[:,:,:y.shape[-2],:y.shape[-1]]+.11;q=y.round();frames.append((q,hs,zs[0],list(z.shape[-2:])))
        return frames
    @torch.no_grad()
    def compress_video(self,x,path,model_id):
        items=[];payloads=[];prev=None;fixed=self.fixed_sequence(x)
        for t,(q,hs,zstream,zshape) in enumerate(fixed):
            start=time.perf_counter();mean,scale=self.params_for(hs,prev);context=time.perf_counter()-start;indexes=self.base.gc.build_indexes(scale);start=time.perf_counter();main=self.base.gc.compress(q,indexes,means=mean)[0];coding=time.perf_counter()-start
            recovered=self.base.gc.decompress([main],indexes,means=mean);assert torch.equal(recovered,q)
            payloads.extend([main,zstream]);items.append(dict(main=len(main),hyper=len(zstream),yshape=list(q.shape[-2:]),zshape=zshape,context_seconds=context,entropy_encode_seconds=coding));prev=self.previous(recovered)
        # Timing is external measurement and must not change stream overhead.
        header=dict(version=1,mode=self.mode,model_id=model_id,hw=list(x.shape[-2:]),items=[{k:v for k,v in item.items() if k not in ('context_seconds','entropy_encode_seconds')} for item in items],sha256=hashlib.sha256(b''.join(payloads)).hexdigest())
        h=json.dumps(header,sort_keys=True,separators=(',',':')).encode();Path(path).write_bytes(b'TPE1'+struct.pack('<I',len(h))+h+b''.join(payloads))
        return dict(actual_bytes=Path(path).stat().st_size,metadata_bytes=len(h)+8,slices=items,sha256=sha(path))
    @torch.no_grad()
    def decompress_video(self,path,model_id):
        b=Path(path).read_bytes()
        if len(b)<8 or b[:4]!=b'TPE1':raise ValueError('magic')
        n=struct.unpack('<I',b[4:8])[0]
        if n>65536 or n>len(b)-8:raise ValueError('header')
        h=json.loads(b[8:8+n]);p=b[8+n:]
        if h['mode']!=self.mode or h['model_id']!=model_id or h['sha256']!=hashlib.sha256(p).hexdigest():raise ValueError('model/checksum')
        if sum(i['main']+i['hyper'] for i in h['items'])!=len(p):raise ValueError('length')
        qvalues=[];recs=[];prev=None;off=0
        for item in h['items']:
            main=p[off:off+item['main']];off+=item['main'];zs=p[off:off+item['hyper']];off+=item['hyper'];zh=self.base.eb.decompress([zs],item['zshape']);hs=self.base.hs(zh)[:,:,:item['yshape'][0],:item['yshape'][1]]+.11;mean,scale=self.params_for(hs,prev);q=self.base.gc.decompress([main],self.base.gc.build_indexes(scale),means=mean);rec=self.base.gs(q)[:,:,:h['hw'][0],:h['hw'][1]];qvalues.append(q);recs.append(rec);prev=self.previous(q)
        return torch.cat(qvalues),torch.cat(recs)
def load_temporal(path):
    c=torch.load(path,weights_only=False,map_location='cpu');m=Temporal(c['config'],c['mode']);restore_state(m,c['state']);return m.eval(),c
if __name__=='__main__':
    m,c=load_temporal(sys.argv[1]);q,z=m.decompress_video(sys.argv[2],sha(sys.argv[1]));torch.save(dict(q=q,z=z),sys.argv[3])
