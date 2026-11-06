#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Download official JM19.0 zip separately, keeping TLS verification enabled.
# https://iphome.hhi.de/suehring/tml/download/jm19.0.zip
if [ ! -d linux/JM ]; then unzip -q linux/jm19.0.zip -d linux; fi
cp JM-backend/source/macroblock.c JM-backend/source/state_io.c linux/JM/lencod/src/
cp JM-backend/source/state_io.h linux/JM/lencod/inc/
python3 - <<'PY'
from pathlib import Path
import re
p=Path('linux/JM/lencod/src/slice.c');s=p.read_text()
if 'jm_state_replay_mb(currMB)' not in s:
    s=s.replace('#include "slice.h"','#include "slice.h"\n#include "state_io.h"')
    pattern=r'      currSlice->encode_one_macroblock \(currMB\);\s+end_encode_one_macroblock\(currMB\);\s+write_macroblock \(currMB, 1\);'
    replacement='''      if (jm_state_mode() == JM_STATE_REPLAY) {
        jm_state_replay_mb(currMB);
        write_macroblock (currMB, 1);
        jm_state_validate_replay_mb(currMB);
      } else {
        currSlice->encode_one_macroblock (currMB);
        end_encode_one_macroblock(currMB);
        write_macroblock (currMB, 1);
        jm_state_dump_mb(currMB);
      }'''
    s,n=re.subn(pattern,replacement,s);assert n==1;p.write_text(s)
PY
make -C linux/JM -j4 CFLAGS=-fcommon > linux/build.log 2>&1
