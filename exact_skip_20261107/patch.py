"""Convert equivalent zero-CBP P16 blocks to Skip without changing pixels."""
import argparse
from pathlib import Path

HELPER = r"""
static unsigned long exact_skip_count;
static int exact_skip_enabled = -1;

static void exact_skip_finish(void)
{
  printf("B1 exact Skip: converted=%lu\n", exact_skip_count);
}

static void exact_skip(Macroblock *mb)
{
  Slice *slice = mb->p_Slice;
  MotionVector p16, skip;
  int i;
  if (exact_skip_enabled < 0)
  {
    exact_skip_enabled = getenv("JM_REPLAY_EXACT_SKIP") != NULL;
    if (exact_skip_enabled)
      atexit(exact_skip_finish);
  }
  if (!exact_skip_enabled || mb->mb_type != 1 || mb->cbp != 0)
    return;
  p16 = slice->all_mv[LIST_0][0][1][0][0];
  FindSkipModeMotionVector(mb);
  skip = slice->all_mv[LIST_0][0][0][0][0];
  if (p16.mv_x != skip.mv_x || p16.mv_y != skip.mv_y)
    return;
  ++exact_skip_count;
  mb->mb_type = mb->best_mode = PSKIP;
  mb->skip_flag = 1;
  for (i=0; i<4; ++i)
    mb->b8x8[i].mode = PSKIP;
}

"""


def patch(root):
    path = root/"lencod/src/state_io.c"
    source = path.read_text()
    if "static void exact_skip(" in source:
        return
    marker = "void jm_state_replay_mb(Macroblock *currMB)"
    call = "    g_b1_residual_active = 0;\n    normalize_coefficients_for_state(currMB);"
    if any(source.count(key)!=1 for key in (marker,call)):
        raise ValueError("Unsupported latent-codec JM source")
    source = source.replace(marker,HELPER+marker)
    source = source.replace(call,call.replace("    normalize_", "    exact_skip(currMB);\n    normalize_"))
    path.write_text(source)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm",type=Path)
    patch(parser.parse_args().jm)
