"""Add an optional final-latent MV center to JM's native full-pel search."""
import argparse
from pathlib import Path

HELPER = r"""
/* Pilot input: little-endian qpel MV pairs for P frames 1..16, raster order. */
#define PILOT_P_FRAMES 16
#define PILOT_MB_ROWS 30
#define PILOT_MB_COLS 52
static short pilot_mvs[PILOT_P_FRAMES][PILOT_MB_ROWS][PILOT_MB_COLS][2];
static int pilot_prior_loaded = 0;
static unsigned long pilot_guided_blocks = 0;

static void pilot_print_counts(void)
{
  fprintf(stdout, "Latent-guided P16 search calls: %lu\n", pilot_guided_blocks);
}

static int pilot_center(Macroblock *mb, MEBlock *block, int range, MotionVector *mv)
{
  const char *path;
  int frame, row, col, limit;
  FILE *handle;
  if (pilot_prior_loaded == 0)
  {
    path = getenv("JM_LATENT_MV_PRIOR");
    if (!path || !*path)
      pilot_prior_loaded = -1;
    else
    {
      handle = fopen(path, "rb");
      if (!handle || fread(pilot_mvs, 1, sizeof(pilot_mvs), handle) != sizeof(pilot_mvs)
          || fgetc(handle) != EOF)
      {
        fprintf(stderr, "Invalid JM_LATENT_MV_PRIOR input\n");
        exit(EXIT_FAILURE);
      }
      fclose(handle);
      pilot_prior_loaded = 1;
      atexit(pilot_print_counts);
    }
  }
  if (pilot_prior_loaded < 0 || mb->p_Slice->slice_type != P_SLICE
      || block->blocktype != 1 || block->list != 0 || block->ref_idx != 0)
    return 0;
  frame = mb->p_Vid->frame_no - 1;
  row = block->pos_y / 16;
  col = block->pos_x / 16;
  if (mb->p_Vid->width != PILOT_MB_COLS * 16 || mb->p_Vid->height != PILOT_MB_ROWS * 16
      || frame < 0 || frame >= PILOT_P_FRAMES || row < 0 || row >= PILOT_MB_ROWS
      || col < 0 || col >= PILOT_MB_COLS || range > 32)
  {
    fprintf(stderr, "Latent MV prior requires the fixed 832x480 1I+16P pilot\n");
    exit(EXIT_FAILURE);
  }
  /* Limit the proposal plus the local window to the pilot's +/-32 px envelope. */
  limit = 32 - range;
  mv->mv_x = 4 * iClip3(-limit, limit, (pilot_mvs[frame][row][col][0] +
      (pilot_mvs[frame][row][col][0] < 0 ? -2 : 2)) / 4);
  mv->mv_y = 4 * iClip3(-limit, limit, (pilot_mvs[frame][row][col][1] +
      (pilot_mvs[frame][row][col][1] < 0 ? -2 : 2)) / 4);
  ++pilot_guided_blocks;
  return 1;
}

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jm_root", type=Path)
    args = parser.parse_args()
    file = args.jm_root / "lencod/src/me_fullsearch.c"
    source = file.read_text()
    if "static int pilot_center(" not in source:
        marker = "// Functions\n"
        start = "  center.mv_x      = mv_block->pos_x_padded + mv->mv_x;"
        if source.count(marker) != 1 or source.count(start) != 1:
            raise ValueError("Unsupported JM source")
        source = source.replace(marker, HELPER + marker)
        source = source.replace(start, "  if (pilot_center(currMB, mv_block, search_range, mv))\n"
                                       "    min_mcost = DISTBLK_MAX;\n" + start)
        file.write_text(source)
    source = file.read_text()
    if "static int pilot_extra_mode(" not in source:
        source = source.replace("static unsigned long pilot_guided_blocks = 0;",
            """static unsigned long pilot_guided_blocks = 0;
static unsigned long pilot_accepted = 0;

static int pilot_extra_mode(void)
{
  static int enabled = -1;
  if (enabled < 0)
    enabled = getenv("JM_LATENT_MV_EXTRA") != NULL;
  return enabled;
}""")
        marker = '  fprintf(stdout, "Latent-guided P16 search calls: %lu\\n", pilot_guided_blocks);'
        assert source.count(marker) == 1
        source = source.replace(marker, marker +
            '\n  fprintf(stdout, "Latent MV extra candidate accepted: %lu\\n", pilot_accepted);')
        source = source.replace("if (pilot_center(currMB, mv_block, search_range, mv))",
                                "if (!pilot_extra_mode() && pilot_center(currMB, mv_block, search_range, mv))")
        point = "  return min_mcost;\n}"
        assert point in source
        source = source.replace(point, r"""  if (pilot_extra_mode())
  {
    MotionVector proposal = *mv;
    if (pilot_center(currMB, mv_block, search_range, &proposal) &&
        (proposal.mv_x != mv->mv_x || proposal.mv_y != mv->mv_y))
    {
      cand.mv_x = mv_block->pos_x_padded + proposal.mv_x;
      cand.mv_y = mv_block->pos_y_padded + proposal.mv_y;
      mcost = mv_cost(p_Vid, lambda_factor, &cand, &pred);
      if (check_for_00 && cand.mv_x == mv_block->pos_x_padded && cand.mv_y == mv_block->pos_y_padded)
      {
        distblk discount = weighted_cost(lambda_factor, 16);
        mcost = mcost > discount ? mcost - discount : 0;
      }
      if (mcost < min_mcost)
      {
        mcost += mv_block->computePredFPel(ref_picture, mv_block, min_mcost-mcost, &cand);
        if (mcost < min_mcost)
        {
          *mv = proposal;
          min_mcost = mcost;
          ++pilot_accepted;
        }
      }
    }
  }
  return min_mcost;
}""", 1)
        file.write_text(source)

    state = args.jm_root / "lencod/src/state_io.c"
    source = state.read_text()
    if "JM_LIVE_RESIDUAL" not in source:
        marker = '    state_fail("B1 residual request is outside the macroblock payload");'
        if source.count(marker) != 1:
            raise ValueError("Unsupported JM state implementation")
        source = source.replace(marker, marker + """
  /* Use JM's native residual computation against the current reconstructed DPB. */
  if (currMB->mb_type != PSKIP && getenv("JM_LIVE_RESIDUAL"))
    return 0;
""")
        state.write_text(source)



if __name__ == "__main__":
    main()
