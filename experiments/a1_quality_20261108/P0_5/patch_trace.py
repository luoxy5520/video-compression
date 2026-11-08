"""Read-only diagnostics for a private copy of the P0 replay encoder."""
import argparse
from pathlib import Path

DETAIL = r'''
  {
    static FILE *p05_bits;
    const char *name = getenv("JM_P05_BITS");
    if (name && currSlice->slice_type == P_SLICE)
    {
      if (!p05_bits) p05_bits = fopen(name, "w");
      if (!p05_bits || fprintf(p05_bits, "%d,%d,%d,%d,%d,%d,%d,%d\n",
          p_Vid->frame_no, currMB->mbAddrX, currMB->mb_type,
          mbBits->mb_total, mbBits->mb_mode, mbBits->mb_inter,
          mbBits->mb_y_coeff, mbBits->mb_uv_coeff + mbBits->mb_cb_coeff + mbBits->mb_cr_coeff) < 0)
        exit(EXIT_FAILURE);
    }
  }
'''

NATIVE = r'''
static void p05_trace(Macroblock *mb)
{
  static FILE *fp;
  const char *name = getenv("JM_P05_MC");
  const char *probe = getenv("JM_P05_SKIP_PROBE");
  Slice *sl = mb->p_Slice;
  PicMotionParams **motion = mb->p_Vid->enc_picture->mv_info;
  MotionVector mv = motion[mb->block_y][mb->block_x].mv[LIST_0];
  MotionVector **skip_slot = sl->all_mv[LIST_0][0][PSKIP];
  MotionVector saved_mv[4][4];
  imgpel saved[3][16][16];
  uint32_t ids[4] = {mb->p_Vid->frame_no, mb->mbAddrX, mb->mb_type, 0};
  int32_t fields[4] = {mv.mv_x, mv.mv_y, 0, 0};
  int p,y,x,b,k,nz=0, uv_nz=0;
  char ref[2] = {0, -1};
  int modes[2] = {PSKIP, PSKIP};
  if (!name) return;
  if (!fp) fp = fopen(name, "wb");
  if (!fp) state_fail("P0.5 MC trace open");
  if (mb->mb_type != PSKIP)
  {
    for (b=0; b<BLOCK_SIZE+mb->p_Vid->num_blk8x8_uv; ++b)
      for (k=0; k<BLOCK_SIZE; ++k)
      {
        int n=0;
        while (n<65 && sl->cofAC[b][k][0][n]) ++n;
        if (b<BLOCK_SIZE) nz += n; else uv_nz += n;
      }
    if (mb->cbp & 0x30)
      for (b=1; b<3; ++b)
        for (k=0; k<18 && sl->cofDC[b][0][k]; ++k) ++uv_nz;
  }
  fields[2]=nz; fields[3]=uv_nz;
  if (fwrite(ids,4,4,fp)!=4 || fwrite(fields,4,4,fp)!=4) state_fail("P0.5 MC trace header");
  for (p=0; p<3; ++p)
    for (y=0; y<(p?8:16); ++y)
    {
      int n=p?8:16;
      memcpy(saved[p][y],sl->mb_pred[p][y],n*sizeof(imgpel));
      if (fwrite(saved[p][y],sizeof(imgpel),n,fp)!=(size_t)n) state_fail("P0.5 actual MC trace");
    }
  if (!probe) return;
  for (y=0; y<4; ++y) memcpy(saved_mv[y],skip_slot[y],sizeof(saved_mv[y]));
  FindSkipModeMotionVector(mb);
  fields[0]=skip_slot[0][0].mv_x; fields[1]=skip_slot[0][0].mv_y;
  if (fwrite(fields,4,2,fp)!=2) state_fail("P0.5 Skip MV trace");
  luma_prediction(mb,0,0,16,16,0,modes,ref,0);
  for (p=0; p<2; ++p)
    for (y=0; y<8; y+=4)
      for (x=0; x<8; x+=4)
        chroma_prediction_4x4(mb,p,x,y,0,PSKIP,PSKIP,0,-1,0);
  for (p=0; p<3; ++p)
    for (y=0; y<(p?8:16); ++y)
    {
      int n=p?8:16;
      if (fwrite(sl->mb_pred[p][y],sizeof(imgpel),n,fp)!=(size_t)n) state_fail("P0.5 Skip MC trace");
      memcpy(sl->mb_pred[p][y],saved[p][y],n*sizeof(imgpel));
    }
  for (y=0; y<4; ++y) memcpy(skip_slot[y],saved_mv[y],sizeof(saved_mv[y]));
}
'''


def insert(path, marker, addition):
    data = path.read_bytes()
    newline = "\r\n" if b"\r\n" in data else "\n"
    key = marker.replace("\n", newline).encode()
    if data.count(key) != 1:
        raise ValueError(f"Unexpected source: {path}, {marker}")
    path.write_bytes(data.replace(key, key + addition.replace("\n", newline).encode()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jm", type=Path)
    source = p.parse_args().jm / "lencod/src"
    state = source / "state_io.c"
    if b"JM_P05_MC" in state.read_bytes():
        raise ValueError("Already patched")
    insert(state, '#include "mv_search.h"', '\n#include "mc_prediction.h"')
    marker = "void jm_state_replay_mb(Macroblock *currMB)"
    data = state.read_bytes()
    assert data.count(marker.encode()) == 1
    state.write_bytes(data.replace(marker.encode(), NATIVE.encode() + b"\n" + marker.encode()))
    insert(state, "    a_coeff_trace(currMB);", "\n    p05_trace(currMB);")
    insert(source / "macroblock.c", "  p_Vid->p_Stats->bit_slice += mbBits->mb_total;", DETAIL)


if __name__ == "__main__":
    main()
