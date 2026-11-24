"""Independent decoding requires only numpy and system libzstd."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from codec.lvc import encode,decode
p=argparse.ArgumentParser(); p.add_argument('mode',choices=['encode','decode']); p.add_argument('input'); p.add_argument('output'); p.add_argument('--bits',type=int,choices=[0,2,3,4,5,6,8],default=4); p.add_argument('--metadata'); a=p.parse_args()
if a.mode=='decode': np.save(a.output,decode(a.input))
else:
 x=np.load(a.input,allow_pickle=False); meta=json.loads(Path(a.metadata).read_text()) if a.metadata else {}
 print(encode(x,a.output,dict(bits=a.bits,granularity='channel'),dict(algorithm='zstd'),meta))
