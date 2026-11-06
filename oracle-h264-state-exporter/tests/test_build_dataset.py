import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildDatasetTest(unittest.TestCase):
    def test_convert_p_targets(self):
        shape = (3, 1, 2)
        is_i = np.zeros(shape, dtype=np.uint8)
        is_i[0] = 1
        is_p = np.zeros(shape, dtype=np.uint8)
        is_p[1:] = 1
        is_skip = np.zeros(shape, dtype=np.uint8)
        is_skip[1, 0, 0] = 1
        mb_type = np.full(shape, 10, dtype=np.int16)
        mb_type[1:] = 1
        mb_type[1, 0, 0] = 0
        mv = np.zeros(shape + (2,), dtype=np.int16)
        mv[1:, :, :, 0] = 4
        has_residual = np.zeros(shape, dtype=np.uint8)
        has_residual[1:] = 1
        has_residual[1, 0, 0] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.npz"
            output = Path(temp_dir) / "target.npz"
            np.savez_compressed(
                source,
                is_p=is_p,
                is_i=is_i,
                is_skip=is_skip,
                mb_type=mb_type,
                mv_qpel_mb=mv,
                mv_is_uniform=is_p.copy(),
                has_residual=has_residual,
                residual_y=np.zeros(shape + (16, 16), dtype=np.int16),
                residual_u=np.zeros(shape + (8, 8), dtype=np.int16),
                residual_v=np.zeros(shape + (8, 8), dtype=np.int16),
            )
            summary = MODULE.convert_p_targets(source, output)
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["mode_id"].shape, (2, 1, 2))
                self.assertEqual(result["mv_qpel"].shape, (2, 1, 2, 2))
                self.assertEqual(int(result["mode_id"][0, 0, 0]), 0)
                self.assertEqual(int(result["mode_id"][0, 0, 1]), 1)
            self.assertEqual(summary["skip_macroblocks"], 1)
            self.assertEqual(summary["p16_macroblocks"], 3)

    def test_link_final_targets_and_curriculum_order(self):
        def row(sample_id, step, total_steps=20):
            return {
                "record_id": f"{sample_id}__step-{step}",
                "sample_id": sample_id,
                "step": str(step),
                "total_steps": str(total_steps),
                "width": 832,
                "height": 480,
                "frames": 17,
                "fps": 16.0,
                "qp": 28,
                "mode_preset": "skip-p16",
                "current_label_path": f"labels/{sample_id}__step-{step}.npz",
                "current_oracle_state_path": f"backend/{sample_id}__step-{step}/oracle_state.bin",
                "current_reference_recon_path": f"backend/{sample_id}__step-{step}/reference_recon.yuv",
                "current_target_summary": {"step": step},
            }

        rows = [
            row("sample_a", 5),
            row("sample_a", 20),
            row("sample_b", 5),
            row("sample_b", 20),
        ]
        order = MODULE.link_final_targets(rows)
        self.assertEqual(order, [20, 5])
        early = rows[0]
        self.assertEqual(early["input_step"], 5)
        self.assertEqual(early["target_step"], 20)
        self.assertEqual(early["remaining_steps"], 15)
        self.assertAlmostEqual(early["step_fraction"], 0.25)
        self.assertEqual(early["current_label_path"], "labels/sample_a__step-5.npz")
        self.assertEqual(early["target_label_path"], "labels/sample_a__step-20.npz")
        self.assertEqual(early["label_path"], early["target_label_path"])
        self.assertEqual(
            early["target_oracle_state_path"],
            "backend/sample_a__step-20/oracle_state.bin",
        )


if __name__ == "__main__":
    unittest.main()
