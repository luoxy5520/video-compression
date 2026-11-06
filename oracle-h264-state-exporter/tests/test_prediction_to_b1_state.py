import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prediction_to_b1_state.py"
SPEC = importlib.util.spec_from_file_location("prediction_to_b1_state", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PredictionAdapterTest(unittest.TestCase):
    def test_skip_mv_zero_shortcut_and_median(self):
        motion = np.zeros((3, 3, 2), dtype=np.int16)
        motion[1, 0] = [4, 2]
        motion[0, 1] = [8, 4]
        motion[0, 2] = [12, 6]
        self.assertTrue(np.array_equal(MODULE.derive_skip_mv(motion, 1, 1), [8, 4]))
        motion[1, 0] = [0, 0]
        self.assertTrue(np.array_equal(MODULE.derive_skip_mv(motion, 1, 1), [0, 0]))
        self.assertTrue(np.array_equal(MODULE.derive_skip_mv(motion, 0, 1), [0, 0]))


if __name__ == "__main__":
    unittest.main()
