"""Add an offline frame-boundary residual exchange to a private JM encoder."""
import argparse
from pathlib import Path

HOOK = r'''
/* The offline teacher receives a complete luma residual frame after providing
   the exact reconstructed reference. Strict replay leaves this hook disabled. */
static void p06_teacher(Macroblock *mb)
{
  static int fd = -1;
  static int16_t residual[1560][16][16];
  const char *name = getenv("JM_P06_SOCKET");
  Slice *slice = mb->p_Slice;
  int y;
  if (!name) return;
  if (mb->p_Vid->width != 832 || mb->p_Vid->height != 480 || mb->qp != 28 ||
      mb->mbAddrX >= 1560 || sizeof(imgpel) != 2)
    state_fail("P0.6 requires fixed 832x480 QP28 and 16-bit trace samples");
  if (fd < 0)
  {
    struct sockaddr_un addr;
    memset(&addr,0,sizeof(addr));
    if (strlen(name) >= sizeof(addr.sun_path)) state_fail("P0.6 socket path");
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path,name);
    fd = socket(AF_UNIX,SOCK_STREAM,0);
    if (fd < 0 || connect(fd,(struct sockaddr *)&addr,sizeof(addr)) < 0)
      state_fail("P0.6 teacher socket");
  }
  if (mb->mbAddrX == 0)
  {
    uint32_t frame = mb->p_Vid->frame_no;
    StorablePicture *ref = slice->listX[LIST_0 + mb->list_offset][0];
    p06_exchange(fd,&frame,sizeof(frame),1);
    for (y=0; y<480; ++y)
      p06_exchange(fd,ref->imgY[y],832*sizeof(imgpel),1);
    p06_exchange(fd,residual,sizeof(residual),0);
  }
  if (mb->mb_type != PSKIP)
    memcpy(g_b1_residual[PLANE_Y],residual[mb->mbAddrX],sizeof(residual[0]));
}
'''

EXCHANGE = r'''
static void p06_exchange(int fd, void *buffer, size_t size, int output)
{
  unsigned char *p = buffer;
  while (size)
  {
    ssize_t n = output ? write(fd,p,size) : read(fd,p,size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) state_fail("P0.6 teacher exchange");
    p += n;
    size -= n;
  }
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jm", type=Path)
    path = p.parse_args().jm / "lencod/src/state_io.c"
    data = path.read_text()
    head = "#include <string.h>"
    marker = "void jm_state_replay_mb(Macroblock *currMB)"
    call = "    clear_all_coefficients(currMB);"
    assert all(data.count(s) == 1 for s in (head, marker, call)) and "JM_P06_SOCKET" not in data
    data = data.replace(head, head + "\n#include <errno.h>\n#include <unistd.h>\n#include <sys/socket.h>\n#include <sys/un.h>")
    data = data.replace(marker, EXCHANGE + HOOK + "\n" + marker)
    data = data.replace(call, "    p06_teacher(currMB);\n" + call)
    path.write_text(data)


if __name__ == "__main__":
    main()
