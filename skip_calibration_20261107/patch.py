"""Allow B1 replay to certify Skip against its active reconstructed reference."""
import argparse
from pathlib import Path

HELPER = r"""
static long skip_gate_threshold = -2;
static unsigned long skip_gate_total, skip_gate_accepted;
static FILE *skip_gate_trace;

static void skip_gate_finish(void)
{
  printf("B1 Skip gate: accepted=%lu total=%lu\n", skip_gate_accepted, skip_gate_total);
  if (skip_gate_trace && fclose(skip_gate_trace))
    fprintf(stderr, "Could not close Skip gate trace\n");
}

static int skip_gate(Macroblock *mb)
{
  Slice *slice = mb->p_Slice;
  VideoParameters *vid = mb->p_Vid;
  DecodedPictureBuffer *dpb = vid->p_Dpb_layer[vid->dpb_layer_id];
  int mode[2] = {0, 0};
  char ref[2] = {0, 0};
  int x, y, p, difference, accept;
  uint32_t error[3] = {0, 0, 0};
  if (skip_gate_threshold == -2)
  {
    const char *value = getenv("JM_SKIP_GATE_MSE");
    const char *trace = getenv("JM_SKIP_GATE_TRACE");
    char *end;
    skip_gate_threshold = value ? strtol(value, &end, 10) : -1;
    if (value && (*end || end == value || skip_gate_threshold < 0 || skip_gate_threshold > 65535
        || !getenv("JM_REPLAY_SYNC_MV") || !getenv("JM_ALIGNED_PROXY")))
      state_fail("invalid Skip gate configuration");
    if (trace && !(skip_gate_trace = fopen(trace, "wb")))
      state_fail("cannot open Skip gate trace");
    if (value)
      atexit(skip_gate_finish);
  }
  if (skip_gate_threshold < 0)
    return 0;
  if (mb->mb_type != 1)
    state_fail("Skip gate requires a P16 candidate for every P block");
  aligned_init();
  ++skip_gate_total;
  FindSkipModeMotionVector(mb);
  dpb->pf_luma_prediction(mb, 0, 0, 16, 16, 0, mode, ref, 0);
  for (p=0; p<2; ++p)
    for (y=0; y<8; y+=4)
      for (x=0; x<8; x+=4)
        chroma_prediction_4x4(mb, p, x, y, 0, 0, 0, 0, 0, 0);
  for (p=0; p<3; ++p)
  {
    int size = p ? 8 : 16;
    int stride = p ? 416 : 832;
    int plane = p ? 832*480 + (p-1)*416*240 : 0;
    int base = vid->frame_no*ALIGNED_FRAME_BYTES + plane + (mb->mbAddrX / 52)*size*stride
               + (mb->mbAddrX % 52)*size;
    for (y=0; y<size; ++y)
      for (x=0; x<size; ++x)
      {
        difference = aligned_source[base+y*stride+x] - slice->mb_pred[p][y][x];
        error[p] += difference*difference;
      }
  }
  accept = (error[0] <= 256*skip_gate_threshold && error[1] <= 64*skip_gate_threshold
            && error[2] <= 64*skip_gate_threshold);
  if (skip_gate_trace)
  {
    uint32_t record[6] = {vid->frame_no, mb->mbAddrX, error[0], error[1], error[2], accept};
    if (fwrite(record, sizeof(record), 1, skip_gate_trace) != 1)
      state_fail("cannot write Skip gate trace");
  }
  if (accept)
  {
    ++skip_gate_accepted;
    mb->mb_type = mb->best_mode = PSKIP;
    mb->skip_flag = 1;
    for (p=0; p<4; ++p)
      mb->b8x8[p].mode = PSKIP;
    sync_replay_motion(mb);
  }
  return accept;
}

"""


def patch(root):
    file = root/"lencod/src/state_io.c"
    source = file.read_text()
    if "static int skip_gate(" in source:
        return
    header = '#include "mv_search.h"'
    function = "void jm_state_replay_mb(Macroblock *currMB)"
    call = "    clear_all_coefficients(currMB);"
    if any(source.count(marker) != 1 for marker in (header, function, call)):
        raise ValueError("Start from the synchronized direct-replay JM")
    source = source.replace(header, header+'\n#include "mc_prediction.h"')
    source = source.replace(function, HELPER+function)
    source = source.replace(call, "    (void)skip_gate(currMB);\n"+call)
    file.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm", type=Path)
    patch(parser.parse_args().jm)
