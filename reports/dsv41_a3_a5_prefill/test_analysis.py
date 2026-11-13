import gzip
import json
import unittest

from analyze import HERE, merge, union, window_metrics


def load_report():
    path = HERE / "data/report_data.json"
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(HERE / "data/report_data.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_report()
        cls.ranks = {r["id"]: r for r in cls.data["ranks"]}

    def test_union_does_not_double_count_parallel_tasks(self):
        self.assertEqual(merge([(0, 4), (2, 8), (8, 10), (20, 21)]), [[0, 10], [20, 21]])
        self.assertEqual(union([(0, 10), (1, 9), (4, 15)]), 15)

    def test_compute_communication_and_free_conserve_wall_time(self):
        def row(s, e, comm=False):
            return dict(s=s, e=e, comm=comm, core="COMMUNICATION" if comm else "AI_CORE")

        m = window_metrics([row(0, 10), row(5, 15, True), row(20, 25)])
        self.assertEqual(m["compute_us"], 15)
        self.assertEqual(m["comm_only_us"], 5)
        self.assertEqual(m["overlap_us"], 5)
        self.assertEqual(m["free_us"], 5)

    def test_confirmed_archive_identity_and_coverage(self):
        self.assertEqual(len(self.ranks), 28)
        self.assertEqual(len(self.data["ranksteps"]), 122)
        a5 = self.ranks["a5_bs12-dp0-tp0"]
        self.assertEqual(a5["npu"][0][1], "Ascend950")
        self.assertEqual(a5["profiler_info"]["cann_version"], "9.3.0")
        self.assertIn("model_runner_v1.py", a5["steps"][2]["execute_scopes"][0]["name"])
        self.assertEqual(len([r for r in self.ranks.values() if r["dataset"] == "a3_recent"]), 16)
        self.assertEqual(len([r for r in self.ranks.values() if r["dataset"] == "a5_bs12"]), 8)

    def test_main_layers_and_timing_conservation(self):
        for rank in self.ranks.values():
            for step in rank["steps"]:
                self.assertEqual(step["correlation_coverage"]["hcpre_main"], 80)
                self.assertEqual(next(o["count"] for o in step["operators"] if o["type"] == "MegaMoe"), 40)
                self.assertEqual(len(step["layers"]), 43)
                m = step["metrics"]
                self.assertAlmostEqual(m["compute_us"] + m["comm_only_us"] + m["free_us"], m["span_us"], places=5)

    def test_local_index_is_not_global_round(self):
        first = self.ranks["a5_bs12-dp5-tp0"]["steps"][0]
        self.assertEqual(first["global_round"], 1)
        groups = self.data["global_rounds"]["a5_bs12"]
        self.assertEqual(groups[0]["rank_count"], 7)
        self.assertEqual(groups[-1]["rank_count"], 3)
        self.assertEqual(self.ranks["a5_bs12-dp0-tp0"]["steps"][2]["round_rank_count"], 8)

    def test_a5_fusion_is_present_and_cpp_attention_is_counted(self):
        step = self.ranks["a5_bs12-dp0-tp0"]["steps"][2]
        self.assertEqual(next(o["count"] for o in step["operators"] if o["type"] == "RmsNormDynamicMxQuant"), 40)
        self.assertEqual(sum(o["count"] for o in step["operators"] if "MqsmlaKernel" in o["type"]), 40)
        self.assertTrue(any("maybe_fuse_rms_norm_mx_quant" in p["name"] for p in self.data["fusion_parents"]))
        self.assertIn("DT_FLOAT4_E2M1", self.data["sample_rows"]["a5_bs12"]["MegaMoe"]["Input Data Types"])

    def test_synchronized_finish_is_not_kernel_compute_speed(self):
        rows = [
            m
            for m in self.data["mega_wait"]
            if m["dataset"] == "a3_recent" and m["round"] == 0 and m["layer"] == 0 and m["tp"] == 0
        ]
        active = next(x for x in rows if x["dp"] == 1)
        dummy = next(x for x in rows if x["dp"] == 0)
        self.assertLess(abs(active["end_ms"] - dummy["end_ms"]), 0.01)
        self.assertGreater(dummy["duration_ms"] / active["duration_ms"], 80)

    def test_historical_long_host_sync_is_mostly_hidden(self):
        prep = self.ranks["a3_032-dp0-tp0"]["steps"][1]["prep"]
        self.assertGreater(prep["gloo_prev_forward_overlap_us"] / 1000, 365)
        self.assertLess(prep["gloo_outside_prev_forward_us"] / 1000, 5.5)


if __name__ == "__main__":
    unittest.main()
