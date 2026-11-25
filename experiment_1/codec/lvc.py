"""LVC1 standalone CPU codec; no VAE or pixel dependencies."""
import json,struct,hashlib,zlib,ctypes,ctypes.util
from pathlib import Path
import numpy as np

def zstd(data,size=None):
    lib=ctypes.CDLL(ctypes.util.find_library('zstd'))
    for name in ('ZSTD_compressBound','ZSTD_compress','ZSTD_decompress','ZSTD_isError'): getattr(lib,name).restype=ctypes.c_size_t
    cap=lib.ZSTD_compressBound(len(data)) if size is None else size
    out=ctypes.create_string_buffer(cap)
    n=lib.ZSTD_compress(out,cap,data,len(data),3) if size is None else lib.ZSTD_decompress(out,cap,data,len(data))
    if lib.ZSTD_isError(n): raise ValueError('zstd failure')
    return out.raw[:n]

def _bits(bits):
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError('bits must be an integer in [2,8]')

def pack(q,bits):
    _bits(bits)
    q=np.asarray(q)
    if not np.issubdtype(q.dtype,np.integer): raise ValueError('integer symbols required')
    if np.any(q < -(1<<(bits-1))) or np.any(q >= (1<<(bits-1))):
        raise ValueError('symbol overflow')
    u=q.astype('int16').ravel()+(1<<(bits-1))
    return np.packbits(((u[:,None]>>np.arange(bits))&1).astype('uint8').ravel(),bitorder='little').tobytes()

def unpack(data,bits,count):
    _bits(bits)
    if type(count) is not int or count < 0: raise ValueError('invalid symbol count')
    if len(data)!=(count*bits+7)//8: raise ValueError('packed length')
    if count*bits%8 and data[-1]>>(count*bits%8): raise ValueError('nonzero padding')
    b=np.unpackbits(np.frombuffer(data,dtype='uint8'),bitorder='little')[:count*bits].reshape(count,bits)
    return b.astype('int16')@(1<<np.arange(bits,dtype='int16'))-(1<<(bits-1))

def encode(latent,output_path,quant_config,compression_config,video_metadata):
    if 'channel_bits' in quant_config:
        import tempfile
        parts=[]
        with tempfile.TemporaryDirectory() as d:
            for c,bits in enumerate(quant_config['channel_bits']):
                cfg=dict(bits=bits,granularity='channel',percentile=quant_config.get('percentile',100))
                path=Path(d)/str(c); encode(latent[:,c:c+1],path,cfg,compression_config,video_metadata); parts.append(path.read_bytes())
        h=json.dumps(dict(version=1,mixed=True,channel_bits=quant_config['channel_bits'],lengths=list(map(len,parts))),separators=(',',':')).encode()
        blob=b'LVC1'+struct.pack('<I',len(h))+h+b''.join(parts); Path(output_path).write_bytes(blob)
        return dict(total_bytes=len(blob),header_bytes=len(h)+8,payload_bytes=sum(map(len,parts)))
    x=np.ascontiguousarray(latent); cfg=quant_config; bits=cfg.get('bits',0)
    if bits: _bits(bits)
    if x.ndim!=5 or not np.isfinite(x).all(): raise ValueError('finite BCTHW required')
    h=dict(version=1,shape=list(x.shape),original_dtype=x.dtype.str,layout='BCTHW',byte_order='little',video=video_metadata,quantization=cfg,compression=compression_config,zero_point=0)
    if bits:
        axis=(0,2,3,4) if cfg.get('granularity')=='channel' else None
        scale=np.maximum(np.percentile(abs(x.astype('float32')),cfg.get('percentile',100),axis=axis,keepdims=True)/((1<<(bits-1))-1),1e-12).astype('float32')
        h['scale']=scale.tolist(); levels=(1<<(bits-1))-1
        q=np.clip(np.rint(x/scale),-levels,levels).astype('int16'); mode=cfg.get('temporal','none')
        if mode=='closed_loop':
            rec=np.zeros_like(x); interval=cfg.get('refresh',0)
            for t in range(x.shape[2]):
                pred=0 if t==0 or (interval and t%interval==0) else rec[:,:,t-1]
                q[:,:,t]=np.clip(np.rint((x[:,:,t]-pred)/scale[:,:,0]),-levels,levels)
                rec[:,:,t]=pred+q[:,:,t]*scale[:,:,0]
        if mode=='integer_delta':
            q=np.concatenate((q[:,:,:1],np.diff(q,axis=2)),axis=2); raw=q.astype('<i2').tobytes(); h['storage']='delta16'
        else: raw=pack(q,bits); h['storage']='packed'
    else:
        raw=x.tobytes(); h['storage']='raw'; h['original_sha256']=hashlib.sha256(raw).hexdigest()
    h['raw_payload_bytes']=len(raw)
    if compression_config.get('shuffle'):
        item=x.dtype.itemsize; raw=np.frombuffer(raw,dtype='uint8').reshape(-1,item).T.copy().tobytes(); h['shuffle_itemsize']=item
    alg=compression_config['algorithm']; p=zlib.compress(raw) if alg=='zlib' else zstd(raw) if alg=='zstd' else raw
    h.update(payload_bytes=len(p),payload_sha256=hashlib.sha256(p).hexdigest())
    header=json.dumps(h,separators=(',',':'),allow_nan=False).encode(); blob=b'LVC1'+struct.pack('<I',len(header))+header+p
    Path(output_path).write_bytes(blob)
    return dict(total_bytes=len(blob),header_bytes=8+len(header),payload_bytes=len(p))

def decode(input_path):
    blob=Path(input_path).read_bytes()
    if blob[:4]!=b'LVC1': raise ValueError('magic')
    n=struct.unpack('<I',blob[4:8])[0]; h=json.loads(blob[8:8+n]); p=blob[8+n:]
    if h.get('mixed'):
        import tempfile
        arrays=[]; offset=0
        with tempfile.TemporaryDirectory() as d:
            for length in h['lengths']:
                path=Path(d)/str(offset); path.write_bytes(p[offset:offset+length]); arrays.append(decode(path)); offset+=length
        if offset!=len(p): raise ValueError('mixed length')
        return np.concatenate(arrays,axis=1)

    if h['version']!=1 or len(p)!=h['payload_bytes'] or hashlib.sha256(p).hexdigest()!=h['payload_sha256']: raise ValueError('integrity')
    alg=h['compression']['algorithm']; raw=zlib.decompress(p) if alg=='zlib' else zstd(p,h['raw_payload_bytes']) if alg=='zstd' else p
    if len(raw)!=h['raw_payload_bytes']: raise ValueError('length')
    if 'shuffle_itemsize' in h: raw=np.frombuffer(raw,dtype='uint8').reshape(h['shuffle_itemsize'],-1).T.copy().tobytes()
    cfg=h['quantization']; bits=cfg.get('bits',0); shape=h['shape']
    if not bits:
        if hashlib.sha256(raw).hexdigest()!=h['original_sha256']: raise ValueError('raw hash')
        return np.frombuffer(raw,dtype=h['original_dtype']).reshape(shape).copy()
    q=np.frombuffer(raw,dtype='<i2').reshape(shape).astype('int32') if h['storage']=='delta16' else unpack(raw,bits,int(np.prod(shape))).reshape(shape)
    if cfg.get('temporal')=='integer_delta': q=np.cumsum(q,axis=2)
    x=q*np.array(h['scale'],dtype='float32')
    if cfg.get('temporal')=='closed_loop':
        interval=cfg.get('refresh',0)
        for t in range(1,shape[2]):
            if not interval or t%interval: x[:,:,t]+=x[:,:,t-1]
    return x.astype(h['original_dtype'])
