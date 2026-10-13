import argparse
import os
import unittest
from unittest.mock import patch
from deploy.serve_qwen import build_command


def args(devices=None, profile="3090-bf16"):
    return argparse.Namespace(profile=profile, devices=devices, model=None, port=8000, context=None)


class SlurmGPUChecks(unittest.TestCase):
    def test_scheduler_mask_is_preserved(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "4"}):
            command, override = build_command(args())
        self.assertIsNone(override)
        self.assertIn("Qwen/Qwen3-8B", command)

    def test_explicit_physical_mask_rejected_in_job(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "4"}):
            with self.assertRaisesRegex(ValueError, "不要传"):
                build_command(args(devices="0"))

    def test_missing_scheduler_gpu_rejected(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "先申请 GPU"):
                build_command(args())


if __name__ == "__main__":
    unittest.main()
