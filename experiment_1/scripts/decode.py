"""Independent receiver: bitstream and codec only; no source data or VAE import."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from codec.lvc import decode
if __name__=='__main__': np.save(sys.argv[2],decode(sys.argv[1]))
