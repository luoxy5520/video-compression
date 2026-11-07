"""Trace and optionally predict B1 residuals from the active JM reference."""
import argparse
from pathlib import Path

HELPER = r"""
#define ALIGNED_FRAME_BYTES (832 * 480 * 3 / 2)
#define ALIGNED_FEATURES 5
static unsigned char *aligned_source;
static double aligned_weights[3][ALIGNED_FEATURES];
static int aligned_active;

static void aligned_init(void)
{
  const char *source, *weights;
  FILE *file;
  if (aligned_active)
    return;
  aligned_active = -1;
  source = getenv("JM_ALIGNED_PROXY");
  weights = getenv("JM_ALIGNED_WEIGHTS");
  if (!source)
  {
    if (weights)
      state_fail("aligned weights require a source proxy");
    return;
  }
  file = fopen(source, "rb");
  aligned_source = malloc(17 * ALIGNED_FRAME_BYTES);
  if (!file || !aligned_source ||
      fread(aligned_source, ALIGNED_FRAME_BYTES, 17, file) != 17 || fgetc(file) != EOF)
    state_fail("invalid aligned source proxy");
  fclose(file);
  if (weights)
  {
    file = fopen(weights, "rb");
    if (!file || fread(aligned_weights, sizeof(aligned_weights), 1, file) != 1 || fgetc(file) != EOF)
      state_fail("invalid aligned head weights");
    fclose(file);
  }
  aligned_active = 1;
}

static int aligned_residual(Macroblock *mb, int plane, int x, int y)
{
  int size = plane == 0 ? 16 : 8;
  int stride = plane == 0 ? 832 : 416;
  int offset = plane == 0 ? 0 : 832 * 480 + (plane - 1) * 416 * 240;
  int frame = mb->p_Vid->frame_no;
  int col = mb->mbAddrX % 52, row = mb->mbAddrX / 52;
  int index = frame * ALIGNED_FRAME_BYTES + offset + (row * size + y) * stride + col * size + x;
  int proxy, mc, old;
  double *w = aligned_weights[plane], correction, result;
  if (mb->p_Vid->width != 832 || mb->p_Vid->height != 480 || frame < 1 || frame > 16 ||
      row < 0 || row >= 30 || x >= size || y >= size)
    state_fail("aligned head requires fixed 832x480, 1I+16P replay");
  proxy = aligned_source[index];
  mc = mb->p_Slice->mb_pred[plane][y][x];
  old = g_b1_residual[plane][y][x];
  correction = w[0] + w[1] * ((proxy - 128) / 128.0) + w[2] * ((proxy - mc) / 128.0)
               + w[3] * (old / 128.0) + w[4] * (abs(proxy - mc) / 128.0);
  result = proxy - mc + correction;
  return iClip3(-255, 255, (int)(result + (result < 0 ? -0.5 : 0.5)));
}

"""


def patch(root):
    path = root / "lencod/src/state_io.c"
    source = path.read_text()
    if source.startswith('#include "mv_search.h"\n'):
        source = source.removeprefix('#include "mv_search.h"\n')
    include = '#include "mbuffer.h"'
    if '#include "mv_search.h"' not in source:
        if source.count(include) != 1:
            raise ValueError("Unsupported B1 source includes")
        source = source.replace(include, include + '\n#include "mv_search.h"', 1)
    if "static void sync_replay_motion(" not in source:
        marker = "void jm_state_replay_mb(Macroblock *currMB)"
        call = "  restore_motion(currMB, &record);"
        if source.count(marker) != 1 or source.count(call) != 1:
            raise ValueError("Unsupported B1 motion restoration")
        source = source.replace(marker, r"""static void sync_replay_motion(Macroblock *mb)
{
  Slice *slice = mb->p_Slice;
  PicMotionParams **motion = mb->p_Vid->enc_picture->mv_info;
  int x, y;
  int mode = mb->mb_type;
  MotionVector **mc = slice->all_mv[LIST_0][0][mode];
  if (mode == PSKIP)
    FindSkipModeMotionVector(mb);
  for (y=0; y<4; ++y)
    for (x=0; x<4; ++x)
    {
      MotionVector *written = &motion[mb->block_y+y][mb->block_x+x].mv[LIST_0];
      if (mode == PSKIP)
        *written = mc[y][x];
      else
        mc[y][x] = *written;
    }
}

""" + marker)
        source = source.replace(call, call + '\n  if (record.slice_type == P_SLICE && getenv("JM_REPLAY_SYNC_MV"))\n'
                                '    sync_replay_motion(currMB);')
    if "static void aligned_init(" in source:
        path.write_text(source)
        return
    old = '''  if (currMB->mb_type != PSKIP && getenv("JM_LIVE_RESIDUAL"))
  {
    trace_live_mc(currMB, plane, block_x, block_y, width, height);
    return 0;
  }'''
    marker = "int jm_state_apply_b1_residual("
    expression = "(currMB->mb_type == PSKIP) ? 0 : (int)g_b1_residual[plane][block_y + y][block_x + x];"
    if any(source.count(pattern) != 1 for pattern in (old, marker, expression)):
        raise ValueError("Start with the previous residual-reference JM source")
    source = source.replace(marker, HELPER + marker).replace(old,
        '''  if (currMB->mb_type != PSKIP)
  {
    trace_live_mc(currMB, plane, block_x, block_y, width, height);
    if (getenv("JM_LIVE_RESIDUAL"))
      return 0;
    aligned_init();
  }''')
    source = source.replace(expression,
        '''(currMB->mb_type == PSKIP) ? 0 :
          aligned_active > 0 ? aligned_residual(currMB, plane, block_x + x, block_y + y) :
          (int)g_b1_residual[plane][block_y + y][block_x + x];''')
    path.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm", type=Path)
    patch(parser.parse_args().jm)
