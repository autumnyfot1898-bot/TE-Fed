"""Failure gates for the generated-data baseline check report."""
from copy import deepcopy
import unittest

from scripts.check_baselines import finite_tree, parse_methods, validate_metrics, validate_summary


class RuntimeCheckReportTests(unittest.TestCase):
    def test_requested_methods_are_complete_distinct_and_known(self):
        registry = {"local": "Local", "fedavg": "FedAvg"}
        self.assertEqual(parse_methods(None, registry), ["local", "fedavg"])
        self.assertEqual(parse_methods(["fedavg"], registry), ["fedavg"])
        for values in (["local", "local"], ["unknown"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_methods(values, registry)

    def test_metric_validation_rejects_missing_view_missing_ap_and_nonfinite_nested_values(self):
        good = {view: {"mAP50": 0.0, "mAP50_95": 0.0, "AP75": 0.0, "num_images": 2,
                       "per_class": {"1": {"mAP": None}}}
                for view in ("participating", "unseen", "new-client", "hard")}
        validate_metrics(good)
        invalid = []
        missing_view = deepcopy(good)
        missing_view.pop("hard")
        invalid.append(missing_view)
        for key, value in (("mAP50", None), ("AP75", float("nan")), ("mAP50_95", 1.1), ("num_images", 1)):
            modified = deepcopy(good)
            modified["hard"][key] = value
            invalid.append(modified)
        nested = deepcopy(good)
        nested["hard"]["per_class"]["1"]["mAP"] = float("inf")
        invalid.append(nested)
        for metrics in invalid:
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                validate_metrics(metrics)

    def test_summary_cannot_pass_when_a_method_failed_was_omitted_or_origin_is_ambiguous(self):
        good = {"data_origin": "generated", "paper_reproduction": False,
                "requested_methods": ["local", "fedavg"],
                "methods": [{"method": "local", "status": "passed"}, {"method": "fedavg", "status": "passed"}]}
        validate_summary(good)
        for mutation in ("failure", "omitted", "wrong_method", "origin", "paper_claim"):
            modified = deepcopy(good)
            if mutation == "failure":
                modified["methods"][1]["status"] = "failed"
            elif mutation == "omitted":
                modified["methods"].pop()
            elif mutation == "wrong_method":
                modified["methods"][1]["method"] = "fedbn"
            elif mutation == "origin":
                modified.pop("data_origin")
            else:
                modified["paper_reproduction"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_summary(modified)

    def test_finite_guard_accepts_optional_values_and_reports_the_nonfinite_path(self):
        finite_tree({"optional": None, "counts": [0, 1], "metric": 0.0})
        with self.assertRaisesRegex(ValueError, r"report.loss\[1\]"):
            finite_tree({"loss": [0.0, float("nan")]}, "report")


if __name__ == "__main__":
    unittest.main()
