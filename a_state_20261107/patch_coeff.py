"""Add an optional read-only coefficient trace to a copy of the synced JM replay."""
import sys
from pathlib import Path

p=Path(sys.argv[1])/"lencod/src/state_io.c"
s=p.read_text()
marker="static void read_coefficients(Macroblock *currMB)"
needle="    g_b1_residual_active = 0;\n    normalize_coefficients_for_state(currMB);"
helper=r'''
static void a_coeff_block(FILE *fp, const int *levels, const int *runs, int cap)
{
  uint8_t count=0;
  int i;
  while (count<cap && levels[count]) ++count;
  if (count>=cap || fwrite(&count,1,1,fp)!=1)
    state_fail("A0 coefficient trace");
  for (i=0;i<count;++i)
  {
    int32_t level=levels[i];
    uint8_t run=(uint8_t)runs[i];
    if (runs[i]<0 || runs[i]>255 || fwrite(&level,4,1,fp)!=1 || fwrite(&run,1,1,fp)!=1)
      state_fail("A0 coefficient trace level/run");
  }
}

static void a_coeff_trace(Macroblock *mb)
{
  static FILE *fp;
  const char *name=getenv("JM_A_COEFF_TRACE");
  Slice *sl=mb->p_Slice;
  uint32_t ids[2]={mb->p_Vid->frame_no,mb->mbAddrX};
  int b8,b4;
  if (!name || mb->mb_type==PSKIP) return;
  if (!fp) fp=fopen(name,"wb");
  if (!fp || fwrite(ids,4,2,fp)!=2) state_fail("A0 coefficient trace IDs");
  for (b8=0;b8<BLOCK_SIZE+mb->p_Vid->num_blk8x8_uv;++b8)
    for (b4=0;b4<BLOCK_SIZE;++b4)
      a_coeff_block(fp,sl->cofAC[b8][b4][0],sl->cofAC[b8][b4][1],65);
  for (b8=0;b8<3;++b8)
    a_coeff_block(fp,sl->cofDC[b8][0],sl->cofDC[b8][1],18);
}

'''
if s.count(marker)!=1 or s.count(needle)!=1 or "static void a_coeff_trace" in s:
    raise ValueError("Unexpected JM tree or patch already present")
p.write_text(s.replace(marker,helper+marker).replace(needle,needle+"\n    a_coeff_trace(currMB);"))
