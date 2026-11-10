"""Protect duration arithmetic and empirical evidence from misleading totals."""

import json
import unittest
from collections import Counter

from analyze import HERE, merge_intervals, percentile, record_kind, union_duration


class TimingTests(unittest.TestCase):
    def test_overlapping_streams_and_parent_child_do_not_double_count(self):
        self.assertEqual(union_duration([(0, 10), (2, 7), (5, 15), (5, 15), (20, 25)]), 20)

    def test_adjacent_and_invalid_intervals(self):
        self.assertEqual(merge_intervals([(10, 20), (0, 10), (5, 5), (30, 20)]), [(0, 20)])
        self.assertEqual(union_duration([]), 0)

    def test_tail_latency_not_hidden_by_mean(self):
        self.assertEqual(percentile([1, 2, 3, 4, 100], 0.5), 3)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 100], 0.95), 80.8)

    def test_communication_parent_and_task_are_distinguished(self):
        self.assertEqual(
            record_kind({"Accelerator Core": "COMMUNICATION", "Name": "hcom_allReduce__0"}),
            "communication_op",
        )
        self.assertEqual(
            record_kind({"Accelerator Core": "COMMUNICATION", "Name": "AivKernel"}),
            "communication_task",
        )

    def test_empirical_windows_support_only_the_claimed_mechanisms(self):
        data = json.loads((HERE / "data/evidence.json").read_text())
        windows = data["engram_windows"]
        self.assertEqual(
            [w["zero_calls"] for w in windows if w["dataset"] == "v030_unlabelled"],
            [1] * 6,
        )
        self.assertEqual([w["zero_calls"] for w in windows if w["dataset"] == "v032_p_tp4"], [0] * 6)
        new_layers = [v for v in data["layers"] if v["dataset"] == "v032_p_tp4"]
        self.assertEqual(Counter(v["tokens"] for v in new_layers), {8188: 63, 128: 57})
        self.assertEqual(len({r["pr"] for r in data["pr_verification"]}), 60)


if __name__ == "__main__":
    unittest.main()
