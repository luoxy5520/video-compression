"""Encode a single fixed-format final latent to a standard H.264 elementary stream."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import torch

from run import OPTIONS, encode_prepared, load_backend, load_models, prepare_one


def main(a):
    if a.qpel and (not a.exact_skip or a.policy!="refine"):
        raise ValueError("The trained qpel head requires --policy refine --exact-skip")
    backend = load_backend(a.kit)
    vae, head = load_models(a)
    a.output.mkdir(parents=True,exist_ok=False)
    prepared = a.output/"prepared"
    timing = prepare_one(a.latent,prepared,vae,head,keep_features=bool(a.qpel))
    extra_env = {}
    qpel_s = 0.0
    if a.qpel:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"learned_qpel_20261107"))
        from qpel_head import model, predict
        network = model().cuda().eval()
        network.load_state_dict(torch.load(a.qpel,weights_only=True))
        qpel_s = predict(network,prepared)
        extra_env.update(JM_QPEL_PRIOR=str(prepared/"qpel_prior.bin"),JM_QPEL_BUDGET=str(a.qpel_budget))
    if a.exact_skip:
        os.environ["JM_REPLAY_EXACT_SKIP"] = "1"
    bitstream, encode_s = encode_prepared(a,backend,prepared,a.output/"encoded",a.policy,extra_env=extra_env)
    log = (a.output/"encoded/encode.log").read_text()
    if a.exact_skip and "B1 exact Skip:" not in log:
        raise ValueError("The selected JM encoder did not implement exact Skip")
    if a.qpel:
        count = re.search(r"B1 latent motion: blocks=(\d+) candidates=(\d+)",log)
        if not count or int(count[1])!=24960 or int(count[2])>24960*(6+2*a.qpel_budget):
            raise ValueError("The selected JM encoder did not apply the learned qpel budget")
    result = dict(bitstream=str(bitstream), policy=a.policy, exact_skip=a.exact_skip, encoder_s=encode_s,
                  qpel_s=qpel_s, qpel_budget=a.qpel_budget if a.qpel else None,
                  latent_to_bitstream_s=timing["preparation_s"]+qpel_s+encode_s,**timing)
    (a.output/"encoding.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("latent","projection","decision","kit","jm","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--policy",choices=OPTIONS,required=True)
    parser.add_argument("--exact-skip",action="store_true")
    parser.add_argument("--qpel",type=Path)
    parser.add_argument("--qpel-budget",type=int,choices=(2,4,6),default=4)
    args = parser.parse_args()
    for key in ("latent","projection","decision","kit","jm","output"):
        setattr(args,key,getattr(args,key).resolve())
    if args.qpel:
        args.qpel = args.qpel.resolve()
    main(args)
