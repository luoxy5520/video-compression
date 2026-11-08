"""Add a read-only CAVLC counter trace to a private JM source copy."""
import argparse
from pathlib import Path

MARKER = "  p_Vid->p_Stats->bit_slice += mbBits->mb_total;"
TRACE = r'''
  /* Offline P0 audit only. No coding choices are changed. */
  {
    static FILE *p0_bits_file;
    const char *p0_bits_path = getenv("JM_P0_MB_BITS");
    if (p0_bits_path && currSlice->slice_type == P_SLICE)
    {
      if (!p0_bits_file)
        p0_bits_file = fopen(p0_bits_path, "w");
      if (!p0_bits_file ||
          fprintf(p0_bits_file, "%d,%d,%d,%d\n", p_Vid->frame_no,
                  currMB->mbAddrX, mbBits->mb_total,
                  mbBits->mb_y_coeff + mbBits->mb_uv_coeff +
                  mbBits->mb_cb_coeff + mbBits->mb_cr_coeff) < 0)
      {
        fprintf(stderr, "Cannot write P0 macroblock bit trace\n");
        exit(EXIT_FAILURE);
      }
    }
  }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jm_copy", type=Path)
    path = parser.parse_args().jm_copy / "lencod/src/macroblock.c"
    data = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    if data.count(MARKER.encode()) != 1 or b"JM_P0_MB_BITS" in data:
        raise ValueError("Unexpected JM source or already patched")
    path.write_bytes(data.replace(MARKER.encode(),
                                  MARKER.encode() + TRACE.replace("\n", newline.decode()).encode()))


if __name__ == "__main__":
    main()
