"""Export a single label-free student state from frozen DiT hidden."""
import argparse
from pathlib import Path

import numpy as np
import torch

from l0 import ARMS, p, predict
from l2 import FREQ, ARMS as L2_ARMS, infer
from repair import to_pixels


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("hidden",type=Path)
    a.add_argument("checkpoint",type=Path)
    a.add_argument("out",type=Path)
    a.add_argument("--arm",choices=[*ARMS,*L2_ARMS],required=True)
    args=a.parse_args()
    torch.set_num_threads(2)
    assert p.digest(p.CKPT)==p.CFG["checkpoint_sha256"]
    frozen=p.load_model(p.CKPT).eval().requires_grad_(False)
    z=p.input_tensor({"hidden_path":str(args.hidden)},"hidden")
    with torch.inference_mode():
        base,_=p.predict(frozen,z,20)
        x=p.feature(frozen,z,base)
        mask=torch.from_numpy(base["mode_id"]==1).cuda()
        freq=ARMS[args.arm][1] if args.arm in ARMS else FREQ
        head=p.SparseHead(67,len(freq)).cuda().eval()
        head.load_state_dict(torch.load(args.checkpoint,weights_only=True))
        if args.arm in ARMS:
            shape=(int(mask.sum()),16,len(freq))
            no_target=torch.empty(shape,device=x.device)
            q,_=predict(head,(x,mask,no_target),freq,64,diagnostics=False)
        else:
            maxima=torch.tensor([p.max_level(f) for f in freq],device=x.device)
            q,_=infer(head,(x,mask,None,None),args.arm,maxima,diagnostics=False)
        base["residual_y"]=to_pixels(q)
    np.savez_compressed(args.out,**base)


if __name__=="__main__":
    main()
