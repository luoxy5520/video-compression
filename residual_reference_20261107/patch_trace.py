"""Export native motion-compensated predictor pixels during live residual replay."""
import argparse
from pathlib import Path

HELPER = r"""
static void trace_live_mc(Macroblock *mb, int plane, int x0, int y0, int w, int h)
{
  static FILE *trace = NULL;
  static int initialized = 0;
  const char *path;
  uint16_t header[7];
  unsigned char row[16];
  int x, y;
  if (!initialized)
  {
    initialized = 1;
    path = getenv("JM_LIVE_MC_TRACE");
    if (path)
    {
      trace = fopen(path, "wb");
      if (!trace)
        state_fail("cannot open MC trace");
    }
  }
  if (!trace)
    return;
  header[0] = (uint16_t)mb->p_Vid->frame_no;
  header[1] = (uint16_t)mb->mbAddrX;
  header[2] = (uint16_t)plane;
  header[3] = (uint16_t)x0;
  header[4] = (uint16_t)y0;
  header[5] = (uint16_t)w;
  header[6] = (uint16_t)h;
  if (fwrite(header, sizeof(header), 1, trace) != 1)
    state_fail("cannot write MC trace header");
  for (y=0; y<h; ++y)
  {
    for (x=0; x<w; ++x)
      row[x] = (unsigned char)mb->p_Slice->mb_pred[plane][y0+y][x0+x];
    if (fwrite(row, 1, w, trace) != (size_t)w)
      state_fail("cannot write MC trace pixels");
  }
}

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jm_root", type=Path)
    args = parser.parse_args()
    path = args.jm_root / "lencod/src/state_io.c"
    source = path.read_text()
    if "static void trace_live_mc(" in source:
        return
    marker = "int jm_state_apply_b1_residual("
    old = '  if (currMB->mb_type != PSKIP && getenv("JM_LIVE_RESIDUAL"))\n    return 0;'
    if source.count(marker)!=1 or source.count(old)!=1:
        raise ValueError("Apply the live residual patch before adding the trace")
    source = source.replace(marker, HELPER+marker).replace(old,
        '  if (currMB->mb_type != PSKIP && getenv("JM_LIVE_RESIDUAL"))\n'
        '  {\n    trace_live_mc(currMB, plane, block_x, block_y, width, height);\n    return 0;\n  }')
    path.write_text(source)


if __name__=="__main__":
    main()
