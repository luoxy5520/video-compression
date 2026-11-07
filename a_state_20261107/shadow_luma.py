"""QP28 luma reference update for restricted JM; no decoder or encoder subprocess."""
import numpy as np

PAT=np.array([[0,1,0,1],[1,2,1,2],[0,1,0,1],[1,2,1,2]])
SCALE=np.array([8192,5243,3355])[PAT]
INV=np.array([16,20,25])[PAT]*16
SCAN=np.array([0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15])
COST=np.array([3,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0])


def fwd(x):
    a,b,c,d=(x[...,i] for i in range(4))
    u,v,w,z=a+d,b+c,b-c,a-d
    return np.stack((u+v,2*z+w,u-v,z-2*w),-1)


def inv(x):
    a,b,c,d=(x[...,i] for i in range(4))
    u,v,w,z=a+c,a-c,(b>>1)-d,b+(d>>1)
    return np.stack((u+z,v+w,v-w,u-z),-1)


def feedback(mc,residual):
    """Prediction + quantized reconstruction, excluding deblocking (disabled)."""
    h,w=mc.shape
    r=residual.astype(np.int32).reshape(h//4,4,w//4,4).transpose(0,2,1,3)
    coef=fwd(fwd(r).swapaxes(-1,-2)).swapaxes(-1,-2)
    lev=((np.abs(coef)*SCALE+(342<<8))>>19)*np.sign(coef)
    ordered=np.abs(lev.reshape(h//4,w//4,16)[...,SCAN])
    cost=np.zeros((h//4,w//4),np.int32)
    last=np.full_like(cost,-1)
    for i in range(16):
        item=ordered[...,i]
        cost+=np.where(item>1,1024,np.where(item==1,COST[(i-last-1).clip(0,15)],0))
        last=np.where(item!=0,i,last)
    cost8=cost.reshape(h//8,2,w//8,2).sum(axis=(1,3))
    used8=cost8>4
    cost16=np.where(used8,cost8,0).reshape(h//16,2,w//16,2).sum(axis=(1,3))
    used8 &= (cost16>5).repeat(2,0).repeat(2,1)
    levels=np.where(used8.repeat(2,0).repeat(2,1)[...,None,None],lev,0)*INV
    rec=inv(inv(levels).swapaxes(-1,-2)).swapaxes(-1,-2)
    rec=((rec+32)>>6).transpose(0,2,1,3).reshape(h,w)
    return (mc.astype(np.int32)+rec).clip(0,255).astype(np.uint8)
