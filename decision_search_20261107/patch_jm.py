"""Add a block shortlist with native cost evaluation and optional wider fallback."""
import argparse
from pathlib import Path

HELPER = r"""
#define DECISION_CANDIDATES 8
#define DECISION_BLOCKS (16 * 30 * 52)
static int16_t decision_input[DECISION_BLOCKS][2 + 2 * DECISION_CANDIDATES];
static int decision_loaded;
static FILE *decision_trace;
static unsigned long decision_narrow, decision_wide, decision_shortlist, decision_skip;

static void decision_close(void)
{
  fprintf(stdout, "Decision budget: narrow=%lu wide=%lu shortlist=%lu skip=%lu\n",
      decision_narrow, decision_wide, decision_shortlist, decision_skip);
  if (decision_trace && fclose(decision_trace))
    exit(EXIT_FAILURE);
}

static const int16_t *decision_row(Macroblock *mb, MEBlock *block, int range)
{
  const char *path;
  FILE *file;
  int index;
  if (!decision_loaded)
  {
    decision_loaded = -1;
    path = getenv("JM_DECISION_SHORTLIST");
    if (path)
    {
      file = fopen(path, "rb");
      if (!file || fread(decision_input, sizeof(decision_input), 1, file) != 1 || fgetc(file) != EOF)
      {
        fprintf(stderr, "Invalid decision shortlist\n");
        exit(EXIT_FAILURE);
      }
      fclose(file);
      decision_loaded = 1;
      path = getenv("JM_DECISION_TRACE");
      if (path && !(decision_trace = fopen(path, "wb")))
        exit(EXIT_FAILURE);
      atexit(decision_close);
    }
  }
  if (decision_loaded < 0 || mb->p_Slice->slice_type != P_SLICE
      || block->blocktype != 1 || block->list != 0 || block->ref_idx != 0)
    return NULL;
  index = (mb->p_Vid->frame_no - 1) * 30 * 52 + mb->mbAddrX;
  if (mb->p_Vid->width != 832 || mb->p_Vid->height != 480 || range != 8
      || index < 0 || index >= DECISION_BLOCKS)
    exit(EXIT_FAILURE);
  return decision_input[index];
}

int decision_try_skip(Macroblock *mb, MEBlock *block)
{
  const int range = imin(block->searchRange.max_x, block->searchRange.max_y) >> 2;
  const int16_t *row = decision_row(mb, block, range);
  return row && (row[0] & 2);
}

void decision_count_skip(void)
{
  ++decision_skip;
}

"""

EXTRA = r"""
  if (decision)
  {
    distblk cheap = max_pos ? min_mcost : DISTBLK_MAX;
    MotionVector cheap_mv = *mv;
    MotionVector native_mv = *mv;
    distblk native_cost = min_mcost;
    int64_t record[7 + DECISION_CANDIDATES];
    int count = decision[1];
    int i;
    if (count < 0 || count > DECISION_CANDIDATES || (!max_pos && !count))
      exit(EXIT_FAILURE);
    for (i = 0; i < count; ++i)
    {
      const int x = decision[2 + 2*i], y = decision[3 + 2*i];
      if (x < -24 || x > 24 || y < -24 || y > 24)
        exit(EXIT_FAILURE);
      cand.mv_x = mv_block->pos_x_padded + 4*x;
      cand.mv_y = mv_block->pos_y_padded + 4*y;
      mcost = mv_cost(p_Vid, lambda_factor, &cand, &pred);
      if (check_for_00 && !x && !y)
      {
        distblk discount = weighted_cost(lambda_factor, 16);
        mcost = mcost > discount ? mcost - discount : 0;
      }
      mcost += mv_block->computePredFPel(ref_picture, mv_block, DISTBLK_MAX-mcost, &cand);
      record[7+i] = mcost;
      if (mcost < cheap)
      {
        cheap = mcost;
        cheap_mv.mv_x = 4*x;
        cheap_mv.mv_y = 4*y;
      }
    }
    ++decision_shortlist;
    if (max_pos <= 9)
    {
      ++decision_narrow;
      *mv = cheap_mv;
      min_mcost = cheap;
    }
    else
      ++decision_wide;
    if (decision_trace)
    {
      record[0] = currMB->p_Vid->frame_no;
      record[1] = currMB->mbAddrX;
      record[2] = native_cost;
      record[3] = native_mv.mv_x;
      record[4] = native_mv.mv_y;
      record[5] = pred_mv->mv_x;
      record[6] = pred_mv->mv_y;
      if (count != DECISION_CANDIDATES || fwrite(record, sizeof(record), 1, decision_trace) != 1)
        exit(EXIT_FAILURE);
    }
  }
"""

MODE_TRACE = r"""
static void trace_decision_mode(Macroblock *mb)
{
  static FILE *output;
  static int initialized;
  const char *path;
  int16_t row[3];
  if (!initialized)
  {
    initialized = 1;
    path = getenv("JM_DECISION_MODE_TRACE");
    if (path && !(output = fopen(path, "wb")))
      exit(EXIT_FAILURE);
  }
  if (!output || mb->p_Slice->slice_type != P_SLICE)
    return;
  row[0] = mb->p_Vid->frame_no;
  row[1] = mb->mbAddrX;
  row[2] = mb->mb_type;
  if (fwrite(row, sizeof(row), 1, output) != 1)
    exit(EXIT_FAILURE);
}

"""

SKIP = r"""
  /* The head proposes Skip; certify luma on the actual decoder reference.
     Keep native residual coding so nonzero coefficients still yield P16. */
  if (!p_Inp->rdopt && decision_try_skip(currMB, mv_block))
  {
    int sx, sy, diff, sse = 0;
    distblk skip_cost;
    FindSkipModeMotionVector(currMB);
    skip_cost = GetSkipCostMB(currMB, lambda_factor[Q_PEL]);
    for (sy = 0; sy < 16; ++sy)
      for (sx = 0; sx < 16; ++sx)
      {
        diff = p_Vid->pCurImg[currMB->opix_y + sy][currMB->pix_x + sx]
             - currSlice->mb_pred[0][sy][sx];
        sse += diff * diff;
      }
    if (sse <= 4 * 256)
    {
      *mv = currSlice->all_mv[0][0][0][0][0];
      min_mcost = skip_cost;
      decision_count_skip();
      goto decision_store_motion;
    }
  }
"""


def replace_once(source, marker, replacement):
    if source.count(marker) != 1:
        raise ValueError(f"Expected one JM source marker: {marker!r}")
    return source.replace(marker, replacement, 1)


def patch(root):
    search = root / "lencod/src/me_fullsearch.c"
    state = root / "lencod/src/state_io.c"
    motion = root / "lencod/src/mv_search.c"
    source, modes, motion_source = search.read_text(), state.read_text(), motion.read_text()
    if "static const int16_t *decision_row" in source:
        return
    source = '#include <stdint.h>\n' + source
    source = replace_once(source, "// Functions\n", HELPER + "// Functions\n")
    marker = "  if (!pilot_extra_mode() && pilot_center(currMB, mv_block, search_range, mv))"
    source = replace_once(source, marker,
        "  const int16_t *decision = decision_row(currMB, mv_block, search_range);\n"
        "  if (decision && (decision[0] & 4))\n    max_pos = 0;\n"
        "  else if (decision && (decision[0] & 1))\n    max_pos = 9;\n" + marker)
    marker = "  if (pilot_extra_mode())\n  {"
    source = replace_once(source, marker, EXTRA + marker)
    marker = "void jm_state_dump_mb(Macroblock *currMB)"
    modes = replace_once(modes, marker, MODE_TRACE + marker)
    marker = "  int derived_cbp;\n\n  if (jm_state_mode() != JM_STATE_DUMP)"
    modes = replace_once(modes, marker,
        "  int derived_cbp;\n\n  trace_decision_mode(currMB);\n  if (jm_state_mode() != JM_STATE_DUMP)")
    marker = "static distblk GetSkipCostMB          (Macroblock *currMB, int lambda);"
    motion_source = replace_once(motion_source, marker, marker +
        "\nextern int decision_try_skip(Macroblock *, MEBlock *);"
        "\nextern void decision_count_skip(void);")
    marker = "  //--- perform motion search ---\n  min_mcost = currMB->IntPelME"
    motion_source = replace_once(motion_source, marker, SKIP + "\n" + marker)
    marker = "  // Set first line\n  for (i=block_x; i < block_x + (bsx>>2); i++)"
    motion_source = replace_once(motion_source, marker, "decision_store_motion:\n" + marker)
    search.write_text(source)
    state.write_text(modes)
    motion.write_text(motion_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm_root", type=Path)
    patch(parser.parse_args().jm_root)
