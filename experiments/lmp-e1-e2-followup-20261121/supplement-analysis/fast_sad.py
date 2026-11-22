import numpy as np
K=(1,-5,20,20,-5,1)
def planes(reference):
 h,w=reference.shape;p=np.pad(reference.astype(np.int32),6,mode='edge');hh=sum(k*p[:,j+1:j+1+w+6] for j,k in enumerate(K));vv=sum(k*p[j+1:j+1+h+6,:] for j,k in enumerate(K));dd=sum(k*hh[j+1:j+1+h+6] for j,k in enumerate(K));return p,((hh+16)>>5).clip(0,255),((vv+16)>>5).clip(0,255),((dd+512)>>10).clip(0,255)
def luma(reference,qpel,cache):
 h,w=reference.shape;yy,xx=np.indices((h,w),dtype=np.int32);mv=qpel.repeat(16,0).repeat(16,1).astype(np.int32);x=xx+mv[...,0]//4;y=yy+mv[...,1]//4;fx,fy=mv[...,0]%4,mv[...,1]%4
 if not (fx.any() or fy.any()):return reference[y.clip(0,h-1),x.clip(0,w-1)]
 p,hh,vv,dd=cache
 def g(kind,sel,dy=0,dx=0):
  sy=(y[sel]+dy).clip(-3,h+2)+3;sx=(x[sel]+dx).clip(-3,w+2)+3
  if kind=='i':return p[sy+3,sx+3]
  if kind=='h':return hh[sy+3,sx]
  if kind=='v':return vv[sy,sx+3]
  return dd[sy,sx]
 out=np.zeros((h,w),np.uint8)
 for j in range(4):
  for i in range(4):
   sel=(fx==i)&(fy==j)
   if not sel.any():continue
   if i%2==0 and j%2==0:val=g({(0,0):'i',(2,0):'h',(0,2):'v',(2,2):'d'}[i,j],sel)
   elif i%2 and j%2:val=(g('h',sel,dy=j//2)+g('v',sel,dx=i//2)+1)//2
   elif i%2:
    a,b=('i','h') if j==0 else ('v','d');val=(g(a,sel,dx=i//2)+g(b,sel)+1)//2
   else:
    a,b=('i','v') if i==0 else ('h','d');val=(g(a,sel,dy=j//2)+g(b,sel)+1)//2
   out[sel]=val
 return out
