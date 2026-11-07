"""Encode a single fixed-format final latent to a standard H.264 elementary stream."""
import argparse
import json
from pathlib import Path

from run import OPTIONS, encode_prepared, load_backend, load_models, prepare_one


def main(a):
    backend = load_backend(a.kit)
    vae, head = load_models(a)
    a.output.mkdir(parents=True,exist_ok=False)
    timing = prepare_one(a.latent,a.output/"prepared",vae,head)
    bitstream, encode_s = encode_prepared(a,backend,a.output/"prepared",a.output/"encoded",a.policy)
    result = dict(bitstream=str(bitstream), policy=a.policy, encoder_s=encode_s,
                  latent_to_bitstream_s=timing["preparation_s"]+encode_s,**timing)
    (a.output/"encoding.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("latent","projection","decision","kit","jm","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--policy",choices=OPTIONS,required=True)
    args = parser.parse_args()
    for key in ("latent","projection","decision","kit","jm","output"):
        setattr(args,key,getattr(args,key).resolve())
    main(args)
