"""Vectorized progressive H.264 4:2:0 MC for one fixed vector per 16x16 block."""
import numpy as np

K=(1,-5,20,20,-5,1)


def luma(reference, qpel):
    h,w=reference.shape
    y,x=np.indices((h,w),dtype=np.int32)
    mv=qpel.repeat(16,0).repeat(16,1).astype(np.int32)
    x=x+(mv[...,0]//4)
    y=y+(mv[...,1]//4)
    fx,fy=mv[...,0]%4,mv[...,1]%4
    # Padded origin is -6. Use the retained -3..W+2 domain for taps.
    pad=np.pad(reference.astype(np.int32),6,mode="edge")
    horizontal=sum(k*pad[:,j+1:j+1+w+6] for j,k in enumerate(K))
    vertical=sum(k*pad[j+1:j+1+h+6,:] for j,k in enumerate(K))
    diagonal=sum(k*horizontal[j+1:j+1+h+6] for j,k in enumerate(K))
    half_h=((horizontal+16)>>5).clip(0,255)
    half_v=((vertical+16)>>5).clip(0,255)
    half_d=((diagonal+512)>>10).clip(0,255)

    def gather(kind,dy=0,dx=0):
        yy=(y+dy).clip(-3,h+2)+3
        xx=(x+dx).clip(-3,w+2)+3
        if kind=="i": return pad[yy+3,xx+3]
        if kind=="h": return half_h[yy+3,xx]
        if kind=="v": return half_v[yy,xx+3]
        return half_d[yy,xx]

    # The interpolation grids above are computed relative to (-3,-3).
    # The half grids on the integer axis have their own array origin below.
    out=np.zeros((h,w),np.uint8)
    for j in range(4):
        for i in range(4):
            sel=(fx==i)&(fy==j)
            if not sel.any():
                continue
            if i%2==0 and j%2==0:
                val=gather({(0,0):"i",(2,0):"h",(0,2):"v",(2,2):"d"}[i,j])
            elif i%2 and j%2:
                val=(gather("h",dy=j//2)+gather("v",dx=i//2)+1)//2
            elif i%2:
                a,b=("i","h") if j==0 else ("v","d")
                val=(gather(a,dx=i//2)+gather(b)+1)//2
            else:
                a,b=("i","v") if i==0 else ("h","d")
                val=(gather(a,dy=j//2)+gather(b)+1)//2
            out[sel]=val[sel]
    return out


def chroma(reference,qpel):
    h,w=reference.shape
    y,x=np.indices((h,w),dtype=np.int32)
    mv=qpel.repeat(8,0).repeat(8,1).astype(np.int32)
    x=x+mv[...,0]//8
    y=y+mv[...,1]//8
    a,b=mv[...,0]%8,mv[...,1]%8
    x0,x1=x.clip(0,w-1),(x+1).clip(0,w-1)
    y0,y1=y.clip(0,h-1),(y+1).clip(0,h-1)
    return (((8-a)*(8-b)*reference[y0,x0]+a*(8-b)*reference[y0,x1]+
             (8-a)*b*reference[y1,x0]+a*b*reference[y1,x1]+32)>>6).astype(np.uint8)
