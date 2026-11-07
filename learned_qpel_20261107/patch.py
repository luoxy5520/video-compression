"""Expose native closed-loop qpel decisions for distillation and ranked replay."""
import argparse
from pathlib import Path

SETUP = r"""
static FILE *qpel_trace;
static unsigned char qpel_order[LATENT_BLOCKS][2][8];
static int qpel_initialized, qpel_budget;

static void qpel_init(void)
{
  const char *path = getenv("JM_QPEL_TRACE");
  const char *prior = getenv("JM_QPEL_PRIOR");
  FILE *file;
  if (path && !(qpel_trace = fopen(path, "wb")))
    state_fail("cannot open qpel teacher trace");
  if (prior)
  {
    const char *budget = getenv("JM_QPEL_BUDGET");
    qpel_budget = budget ? atoi(budget) : 0;
    file = fopen(prior, "rb");
    if (qpel_budget < 1 || qpel_budget > 8 || !file
        || fread(qpel_order, sizeof(qpel_order), 1, file) != 1 || fgetc(file) != EOF)
      state_fail("invalid learned qpel prior");
    fclose(file);
  }
  qpel_initialized = 1;
}

"""


def patch(root):
    path = root/"lencod/src/state_io.c"
    source = path.read_text()
    if "static void qpel_init(" in source:
        return

    def edit(old,new):
        nonlocal source
        if source.count(old)!=1:
            raise ValueError(f"Unsupported shortlist source: {old}")
        source = source.replace(old,new,1)

    edit("static void select_latent_mv(",SETUP+"static void select_latent_mv(")
    edit("  int refine = getenv", "  int labels[2] = {8, 8};\n  int refine = getenv")
    edit("  ++latent_motion_blocks;", """  if (!qpel_initialized)
    qpel_init();
  ++latent_motion_blocks;""")
    edit("int count = pass ? 8 : LATENT_CANDIDATES+2;",
         "int count = pass ? (qpel_budget ? qpel_budget : 8) : LATENT_CANDIDATES+2;")
    edit("        int slot = i+(i>=4);", """        int direction = qpel_budget ? qpel_order[mb_id][pass-1][i] : i;
        int slot = direction+(direction>=4);
        if (direction >= 8)
          state_fail("invalid learned qpel direction");""")
    edit("        best = mv;","""        best = mv;
        if (pass)
          labels[pass-1] = qpel_budget ? qpel_order[mb_id][pass-1][i] : i;""")
    edit("  for (y=0; y<4; ++y)\n    for (x=0; x<4; ++x)\n    {\n      scratch[y][x] = best;", """  if (qpel_trace)
  {
    uint16_t row[4] = {vid->frame_no, mb->mbAddrX, labels[0], labels[1]};
    if (fwrite(row, sizeof(row), 1, qpel_trace) != 1)
      state_fail("cannot write qpel teacher trace");
  }
  for (y=0; y<4; ++y)
    for (x=0; x<4; ++x)
    {
      scratch[y][x] = best;""")
    path.write_text(source)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jm",type=Path)
    patch(parser.parse_args().jm)
