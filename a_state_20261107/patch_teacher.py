"""Read-only export of the actual P state emitted by the learned-qpel teacher."""
import sys
from pathlib import Path

path=Path(sys.argv[1])/"lencod/src/state_io.c"
source=path.read_text()

def edit(old,new):
    global source
    if source.count(old)!=1:
        raise ValueError(f"Unexpected JM source: {old}")
    source=source.replace(old,new,1)

marker="int jm_state_apply_b1_residual("
edit(marker,r'''
static FILE *a_teacher_file;
static int a_teacher_init;
static int16_t a_teacher_res[3][16][16];

static void a_teacher_start(void)
{
  if (!a_teacher_init)
  {
    const char *name=getenv("JM_A_TEACHER_TRACE");
    a_teacher_init=1;
    if (name && !(a_teacher_file=fopen(name,"wb")))
      state_fail("cannot open teacher trace");
  }
  if (a_teacher_file) memset(a_teacher_res,0,sizeof(a_teacher_res));
}

static void a_teacher_emit(Macroblock *mb)
{
  int16_t h[5]={mb->p_Vid->frame_no,mb->mbAddrX,mb->mb_type,0,0};
  int plane,row;
  if (!a_teacher_file) return;
  if (mb->mb_type!=PSKIP)
  {
    MotionVector mv=mb->p_Vid->enc_picture->mv_info[mb->block_y][mb->block_x].mv[LIST_0];
    h[3]=mv.mv_x;
    h[4]=mv.mv_y;
  }
  else
    memset(a_teacher_res,0,sizeof(a_teacher_res));
  if (fwrite(h,sizeof(h),1,a_teacher_file)!=1) state_fail("teacher trace header");
  for (plane=0;plane<3;++plane)
  {
    int n=plane?8:16;
    for (row=0;row<n;++row)
      if (fwrite(a_teacher_res[plane][row],sizeof(int16_t),n,a_teacher_file)!=(size_t)n)
        state_fail("teacher trace residual");
  }
}

'''+marker)
edit('''    for (x = 0; x < width; ++x)
      currSlice->mb_ores[plane][block_y + y][block_x + x] =
        (currMB->mb_type == PSKIP) ? 0 :
          aligned_active > 0 ? aligned_residual(currMB, plane, block_x + x, block_y + y) :
          (int)g_b1_residual[plane][block_y + y][block_x + x];''',
'''    for (x = 0; x < width; ++x)
    {
      int v = (currMB->mb_type == PSKIP) ? 0 :
          aligned_active > 0 ? aligned_residual(currMB, plane, block_x + x, block_y + y) :
          (int)g_b1_residual[plane][block_y + y][block_x + x];
      currSlice->mb_ores[plane][block_y + y][block_x + x] = v;
      if (a_teacher_file) a_teacher_res[plane][block_y + y][block_x + x] = v;
    }''')
edit('    g_b1_residual_active = 1;\n    currMB->p_Slice->luma_residual_coding',
     '    a_teacher_start();\n    g_b1_residual_active = 1;\n    currMB->p_Slice->luma_residual_coding')
edit('    exact_skip(currMB);\n','    exact_skip(currMB);\n    a_teacher_emit(currMB);\n')
path.write_text(source)
