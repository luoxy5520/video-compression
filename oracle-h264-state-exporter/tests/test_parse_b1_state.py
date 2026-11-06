import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "parse_b1_state.py"
SPEC = importlib.util.spec_from_file_location("parse_b1_state", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParseB1StateTest(unittest.TestCase):
    def test_minimal_i_plus_p16_state(self):
        header = MODULE.HEADER.pack(
            MODULE.EXPECTED_MAGIC,
            7,
            16,
            16,
            1,
            6,
            1,
            0,
            66,
            40,
            1,
            2,
            2,
            MODULE.MB_RECORD.size,
            0,
        )
        zero_mv = [0] * 32
        no_ref = [-1] * 16
        i_record = MODULE.MB_RECORD.pack(
            0, 0, MODULE.I_SLICE, 10, 28, 2, 0,
            0, 0, 0, 0,
            *zero_mv,
            *no_ref,
        )
        sparse_blocks = bytes([0] * (6 * 4 + 3))
        i_samples = bytes([64] * 256 + [128] * 64 + [128] * 64)

        mv = [4, -2] * 16
        refs = [0] * 16
        p_record = MODULE.MB_RECORD.pack(
            1, 0, MODULE.P_SLICE, 1, 28, 0, 0,
            1, 1, 1, 1,
            *mv,
            *refs,
        )
        residual = struct.pack("<384h", *([3] * 384))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.bin"
            path.write_bytes(header + i_record + sparse_blocks + i_samples + p_record + residual)
            arrays, metadata = MODULE.parse_state(path)

        self.assertEqual(arrays["mv_qpel"].shape, (2, 1, 1, 4, 4, 2))
        self.assertEqual(arrays["residual_y"].shape, (2, 1, 1, 16, 16))
        self.assertEqual(int(arrays["has_i_recon"][0, 0, 0]), 1)
        self.assertEqual(int(arrays["has_residual"][1, 0, 0]), 1)
        self.assertTrue(np.all(arrays["mv_qpel_mb"][1, 0, 0] == [4, -2]))
        self.assertTrue(np.all(arrays["residual_y"][1, 0, 0] == 3))
        self.assertEqual(metadata["counts"]["records"], 2)


if __name__ == "__main__":
    unittest.main()
