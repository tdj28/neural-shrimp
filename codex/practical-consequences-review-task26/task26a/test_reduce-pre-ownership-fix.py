import copy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("own_reduce", HERE / "reduce.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
spec2 = importlib.util.spec_from_file_location("own_driver", HERE / "run.py")
driver = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(driver)


def fixture():
    def p(carry):
        a, b = (19, 12) if carry else (10, 11)
        return {"a": a, "b": b, "answer": a + b, "carry": carry}
    problems = {"dev": [p(i % 2 == 0) for i in range(8)],
                "heldout": [p(i % 2 == 0) for i in range(96)],
                "distractors": [p(False), p(True), p(False)]}
    records, progress = [], []
    for dtype in r.DTYPES:
        for which in ("dev", "heldout"):
            for i, problem in enumerate(problems[which]):
                for mode, lt in r.MODES:
                    item = {"set": which, "i": i, "problem": problem, "dtype": dtype,
                            "mode": mode, "log10_temperature": lt, "answers_differ": False, "rerun_identical": True}
                    for layout in r.LAYOUTS:
                        item[layout] = {"text": str(problem["answer"]), "parsed": problem["answer"], "correct": True, "margin": 1.0}
                    item["soft_distance" if mode == "soft" else "first_token_difference"] = [1e-6] * 32 if mode == "soft" else None
                    records.append(item)
                progress.append({"dtype": dtype, "set": which, "done": i + 1, "of": len(problems[which]), "seconds": len(progress)})
    for i in range(6):
        for lt in r.TEMPS:
            for delta in r.DELTAS:
                records.append({"set": "mech", "i": i, "problem": problems["heldout"][i], "dtype": "float64",
                                "log10_temperature": lt, "delta": delta, "base_text": "31", "text": "31",
                                "answers_differ": False, "distance": [delta] * 32})
    return records, progress, problems


class TinyControls(unittest.TestCase):
    def test_canonical_is_not_text_equality(self):
        self.assertEqual(r.canonical("answer: -12 later 9"), -12)
        self.assertEqual(r.canonical("0012"), r.canonical("12.0"))
        self.assertIsNone(r.canonical("no integer"))

    def test_zero_contrast_inclusive_p(self):
        x = r.paired_stats([0, 0, 0], 40)
        self.assertEqual(x["ci95_pp"], [0.0, 0.0])
        self.assertEqual(x["signflip_exceedances"], 40)
        self.assertEqual(x["signflip_p_plus_one"], 1.0)

    def test_single_unit_two_sided_ties(self):
        x = r.paired_stats([1], 40)
        self.assertEqual(x["delta_pp"], 50.0)
        self.assertEqual(x["ci95_pp"], [50.0, 50.0])
        self.assertEqual(x["signflip_exceedances"], 40)

    def test_balanced_contrast(self):
        x = r.paired_stats([-2, 2], 100)
        self.assertEqual(x["delta_pp"], 0)
        self.assertEqual(x["ci95_pp"], [-100.0, 100.0])
        self.assertEqual(x["signflip_p_raw"], 1.0)

    def test_outward_signed_rounding(self):
        self.assertEqual(r.outward([-1.21, 2.21]), ["-1.3", "2.3"])

    def test_growth_and_undefined(self):
        self.assertAlmostEqual(r.growth([math.exp(i) for i in range(32)])["value"], 1)
        self.assertIsNone(r.growth([0.0] * 32)["value"])
        self.assertIsNone(r.growth([1.0] * 31 + [0.0])["value"])
        with self.assertRaises(ValueError):
            r.growth([float("nan")] * 32)

    def test_complete_keys_and_invalid_pairs(self):
        rows, progress, problems = fixture()
        target = next(x for x in rows if x["set"] == "heldout" and x["i"] == 0 and x.get("mode") == "soft" and x["dtype"] == "bf16")
        for l in r.LAYOUTS:
            target[l].update(text="INVALID", parsed=None, correct=False)
        result = r.calculate(rows, progress, problems, draws=32)
        self.assertEqual(result["coverage"], {"pipeline_records": 832, "mechanism_records": 54, "progress_records": 208,
                                              "heldout_instances": 96, "development_instances": 8})
        self.assertTrue(result["saved_field_consistency_passed"])
        self.assertEqual(result["conditions"]["heldout"]["bf16:soft:0.0"]["pair_invalid_count"], 1)
        self.assertEqual(result["endpoints"]["bf16:primary_temperatures_mean"]["delta_pp"], 0)

    def test_canonical_primary_and_recorded_mismatch(self):
        rows, progress, problems = fixture()
        target = next(x for x in rows if x["set"] == "heldout" and x["i"] == 0 and x.get("mode") == "soft" and x["dtype"] == "bf16")
        target["batch4"]["text"] = "leading text " + target["single"]["text"]
        target["answers_differ"] = True
        target["batch4"]["correct"] = False
        result = r.calculate(rows, progress, problems, draws=32)
        self.assertEqual(result["endpoints"]["bf16:primary_temperatures_mean"]["sum_half_units"], 0)
        self.assertEqual(result["conditions"]["heldout"]["bf16:soft:0.0"]["text_vs_canonical_disagreement_count"], 1)
        self.assertFalse(result["saved_field_consistency_passed"])
        self.assertEqual(len(result["saved_field_differences"]), 1)

    def test_primary_half_unit_pairing(self):
        rows, progress, problems = fixture()
        target = next(x for x in rows if x["set"] == "heldout" and x["i"] == 0 and x.get("mode") == "soft" and x["dtype"] == "bf16")
        target["batch4"].update(text="999", parsed=999, correct=False)
        target["answers_differ"] = True
        result = r.calculate(rows, progress, problems, draws=32)
        x = result["endpoints"]["bf16:primary_temperatures_mean"]
        self.assertEqual(x["half_units"], [1] + [0] * 95)
        self.assertAlmostEqual(x["delta_pp"], 50 / 96)
        self.assertEqual(result["endpoints"]["float64:primary_temperatures_mean"]["delta_pp"], 0)

    def test_missing_duplicate_and_invalid_key(self):
        rows, progress, problems = fixture()
        for bad in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                r.calculate(bad, progress, problems, draws=32)
        bad = copy.deepcopy(rows)
        bad[0]["log10_temperature"] = False
        with self.assertRaises(ValueError):
            r.calculate(bad, progress, problems, draws=32)

    def test_progress_and_rerun_implication(self):
        rows, progress, problems = fixture()
        bad = copy.deepcopy(progress)
        bad[-1]["done"] = 95
        with self.assertRaises(ValueError):
            r.calculate(rows, bad, problems, draws=32)
        rows[0]["rerun"].update(text="77", parsed=77, correct=False)
        result = r.calculate(rows, progress, problems, draws=32)
        self.assertFalse(result["saved_field_consistency_passed"])
        self.assertEqual(len(result["saved_field_differences"]), 1)

    def test_driver_admission_json_and_pin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                driver.execute({"admission": "prepared"}, root / "out")
            self.assertFalse((root / "out").exists())
            f = root / "x"
            f.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):
                driver.load(f)
            with self.assertRaises(ValueError):
                driver.pin_check({"path": str(f), "bytes": f.stat().st_size, "sha256": "0" * 64})
            self.assertEqual(driver.pointer({"a": [4]}, "/a/0"), 4)


if __name__ == "__main__":
    unittest.main()
