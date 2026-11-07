"""Encode I from the latent proxy and choose P motion from a bounded shortlist."""
import argparse
from pathlib import Path

HELPER = r"""
#define LATENT_BLOCKS (16*30*52)
#define LATENT_CANDIDATES 4
static int16_t latent_mv[LATENT_BLOCKS][LATENT_CANDIDATES][2];
static int latent_mv_loaded;
static unsigned long latent_motion_blocks;
static unsigned long latent_motion_tests;

static void latent_motion_finish(void)
{
  printf("B1 latent motion: blocks=%lu candidates=%lu\n", latent_motion_blocks, latent_motion_tests);
}

static int signed_exp_bits(int delta)
{
  unsigned int code = delta > 0 ? 2*delta : 1-2*delta;
  int bits = 1;
  while (code > 1)
  {
    code >>= 1;
    bits += 2;
  }
  return bits;
}

static void select_latent_mv(Macroblock *mb)
{
  Slice *slice = mb->p_Slice;
  VideoParameters *vid = mb->p_Vid;
  PicMotionParams **motion = vid->enc_picture->mv_info;
  DecodedPictureBuffer *dpb = vid->p_Dpb_layer[vid->dpb_layer_id];
  MotionVector **scratch = slice->all_mv[LIST_0][0][1], predictor, best;
  PixelPos neighbors[4];
  int mode[2] = {1, 0};
  char ref[2] = {0, 0};
  int x, y, i, pass, current, d, base, mb_id, best_cost = INT_MAX;
  int refine = getenv("JM_LATENT_REFINE") != NULL;
  const char *path;
  FILE *file;
  if (!latent_mv_loaded)
  {
    path = getenv("JM_LATENT_SHORTLIST");
    latent_mv_loaded = path ? 1 : -1;
    if (path)
    {
      file = fopen(path, "rb");
      if (!file || fread(latent_mv, sizeof(latent_mv), 1, file) != 1 || fgetc(file) != EOF)
        state_fail("invalid final-latent MV shortlist");
      fclose(file);
      atexit(latent_motion_finish);
    }
  }
  if (latent_mv_loaded < 0)
    return;
  if (mb->mb_type != 1 || !getenv("JM_REPLAY_SYNC_MV") || !getenv("JM_ALIGNED_PROXY"))
    state_fail("shortlist requires synchronized P16 with source proxy");
  aligned_init();
  mb_id = (vid->frame_no-1)*1560 + mb->mbAddrX;
  if (mb_id < 0 || mb_id >= LATENT_BLOCKS)
    state_fail("shortlist outside the configured frame grid");
  base = vid->frame_no*ALIGNED_FRAME_BYTES + mb->pix_y*832 + mb->pix_x;
  get_neighbors(mb, neighbors, 0, 0, 16);
  mb->GetMVPredictor(mb, neighbors, &predictor, 0, motion, LIST_0, 0, 0, 16, 16);
  FindSkipModeMotionVector(mb);
  best = predictor;
  ++latent_motion_blocks;
  for (pass=0; pass<(refine ? 3 : 1); ++pass)
  {
    MotionVector center = best;
    int count = pass ? 8 : LATENT_CANDIDATES+2;
    int step = pass==1 ? 2 : 1;
    for (i=0; i<count; ++i)
    {
      MotionVector mv;
      if (!pass && i<LATENT_CANDIDATES)
      {
        mv.mv_x = latent_mv[mb_id][i][0];
        mv.mv_y = latent_mv[mb_id][i][1];
      }
      else if (!pass)
        mv = i==LATENT_CANDIDATES ? predictor : slice->all_mv[LIST_0][0][0][0][0];
      else
      {
        int slot = i+(i>=4);
        mv.mv_x = center.mv_x+(slot%3-1)*step;
        mv.mv_y = center.mv_y+(slot/3-1)*step;
      }
      if (abs(mv.mv_x)>128 || abs(mv.mv_y)>128)
        continue;
      for (y=0; y<4; ++y)
        for (x=0; x<4; ++x)
          scratch[y][x] = mv;
      dpb->pf_luma_prediction(mb, 0, 0, 16, 16, 0, mode, ref, 0);
      current = 24*(signed_exp_bits(mv.mv_x-predictor.mv_x)+signed_exp_bits(mv.mv_y-predictor.mv_y));
      for (y=0; y<16 && current<best_cost; ++y)
        for (x=0; x<16; ++x)
        {
          d = aligned_source[base+y*832+x]-slice->mb_pred[0][y][x];
          current += d*d;
        }
      ++latent_motion_tests;
      if (current<best_cost)
      {
        best_cost = current;
        best = mv;
      }
    }
  }
  for (y=0; y<4; ++y)
    for (x=0; x<4; ++x)
    {
      scratch[y][x] = best;
      motion[mb->block_y+y][mb->block_x+x].mv[LIST_0] = best;
    }
}

"""


def patch(root):
    state = root/"lencod/src/state_io.c"
    slice_path = root/"lencod/src/slice.c"
    source, slices = state.read_text(), slice_path.read_text()
    if "static void select_latent_mv(" in source:
        return
    marker = "void jm_state_replay_mb(Macroblock *currMB)"
    gate = "    (void)skip_gate(currMB);"
    replay = "      if (jm_state_mode() == JM_STATE_REPLAY) {"
    if source.count(marker)!=1 or source.count(gate)!=1 or slices.count(replay)!=1:
        raise ValueError("Start from the calibrated Skip JM source")
    source = '#include <limits.h>\n'+source
    source = source.replace('#include "mc_prediction.h"', '#include "mc_prediction.h"\n#include "mb_access.h"',1)
    source = source.replace(marker, HELPER+marker)
    source = source.replace(gate,"    if (!skip_gate(currMB))\n      select_latent_mv(currMB);")
    slices = slices.replace(replay, """      if (jm_state_mode() == JM_STATE_REPLAY &&
          !(currSlice->slice_type == I_SLICE && getenv("JM_REPLAY_NATIVE_I"))) {""")
    state.write_text(source)
    slice_path.write_text(slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm", type=Path)
    patch(parser.parse_args().jm)
