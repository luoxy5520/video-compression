import difflib
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = Path.home() / "lmp-e1-e2-followup-20261121/jm-perblock"
jm = root / "jm"
shutil.copytree(base, jm, ignore=shutil.ignore_patterns("obj", "*.exe"))
source = jm / "lencod/src/me_epzs_int.c"
original = source.read_text()
marker = "// Functions\n"
assert original.count(marker) == 1
added = original.replace(marker, marker + (root / "scripts/fixed_search.c.inc").read_text())
marker = "  // Clear EPZSMap"
assert added.count(marker) >= 1
added = added.replace(marker, """  if (blocktype == 1 && list == 0 && ref == 0 &&
      p_Vid->frame_no >= 1 && p_Vid->frame_no <= 16)
  {
    int mode = lmp_mode();
    if (mode >= 4 && mode <= 6)
    {
      min_mcost = lmp_fixed_search(currMB, mv_block, ref_picture, pred,
                                   *mv, mode, lambda_factor, &tmp);
      *prevSad = min_mcost;
      if (p_motion)
        *p_motion = tmp;
      *mv = tmp;
      lmp_reason = 8;
      return min_mcost;
    }
  }

""" + marker, 1)
source.write_text(added)
rel = source.relative_to(jm)
(root / "configs/jm_fixed_search.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(True), added.splitlines(True), f"a/{rel}", f"b/{rel}",
)))
subprocess.run(["make", "-j4", "CFLAGS=-fcommon"], cwd=jm / "lencod",
               check=True)
