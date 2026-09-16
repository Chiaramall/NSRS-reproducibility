import csv
import json
import tempfile
import unittest
from pathlib import Path

from movielens_paper_benchmark.report import write_reports, write_search_audit


def synthetic_summary():
    return {
        "ml-100k": {
            "1.0": {
                "mf-mse": {
                    "metrics": {
                        "hit@10": {"mean": 0.5, "std": 0.1, "n": 4},
                        "ndcg@10": {"mean": 0.25, "std": 0.05, "n": 4},
                    },
                    "training_seconds": {"mean": 2.0, "std": 0.2, "n": 4},
                    "inference_seconds": {"mean": 0.1, "std": 0.01, "n": 4},
                    "memory": {
                        "cpu_process_peak_mib": {
                            "mean": 128.0,
                            "std": 8.0,
                            "n": 4,
                        }
                    },
                    "target_support": {
                        "positive_catalog_items": {
                            "mean": 100.0,
                            "std": 2.0,
                            "n": 4,
                        },
                        "validation_target_positive_support_rate": {
                            "mean": 0.95,
                            "std": 0.01,
                            "n": 4,
                        },
                        "test_target_positive_support_rate": {
                            "mean": 0.90,
                            "std": 0.02,
                            "n": 4,
                        },
                    },
                    "parameters": {"embedding_dim": 64, "biased": True},
                    "n_runs": 4,
                },
                "ltn": {
                    "metrics": {
                        "hit@10": {"mean": 0.55, "std": 0.0, "n": 1},
                        "ndcg@10": {"mean": 0.30, "std": 0.0, "n": 1},
                    },
                    "training_seconds": {"mean": 3.0, "std": 0.0, "n": 1},
                    "inference_seconds": {"mean": 0.2, "std": 0.0, "n": 1},
                    "parameters": {"embedding_dim": 128, "alpha": 0.9},
                    "n_runs": 1,
                },
            },
            "0.2": {
                "mf-mse": {
                    "metrics": {
                        "hit@10": {"mean": 0.2, "std": 0.02, "n": 4},
                        "ndcg@10": {"mean": 0.1, "std": 0.01, "n": 4},
                    },
                    "training_seconds": {"mean": 0.5, "std": 0.05, "n": 4},
                    "inference_seconds": {"mean": 0.1, "std": 0.01, "n": 4},
                    "parameters": {"embedding_dim": 64, "biased": False},
                    "n_runs": 4,
                }
            },
        },
        "ml-1m": {
            "1.0": {
                "ease": {
                    "metrics": {
                        "hit@10": {"mean": 0.6, "std": 0.03, "n": 3},
                        "ndcg@10": {"mean": 0.33, "std": 0.02, "n": 3},
                    },
                    "training_seconds": {"mean": 8.0, "std": 0.4, "n": 3},
                    "inference_seconds": {"mean": 0.5, "std": 0.04, "n": 3},
                    "cpu_process_peak_mib": {
                        "mean": 512.0,
                        "std": 16.0,
                        "n": 3,
                    },
                    "parameters": {"regularization": 500.0},
                    "n_runs": 3,
                }
            }
        },
    }


class ReportTest(unittest.TestCase):
    def test_all_formats_preserve_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(synthetic_summary(), Path(directory))
            self.assertEqual(
                set(paths),
                {
                    "metrics_csv",
                    "resources_csv",
                    "hyperparameters_json",
                    "markdown",
                    "latex",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with paths["metrics_csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            row = next(
                item
                for item in rows
                if item["dataset"] == "ml-100k"
                and item["proportion"] == "1.0"
                and item["model"] == "mf-mse"
                and item["metric"] == "hit@10"
            )
            self.assertAlmostEqual(float(row["std"]), 0.1)
            self.assertEqual(row["n"], "4")

            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("## ml-100k", markdown)
            self.assertIn("0.5000 ± 0.1000 (n=4)", markdown)
            self.assertIn("cpu_process_peak_mib", markdown)

            latex = paths["latex"].read_text(encoding="utf-8")
            self.assertIn("Paper-compatible sampled-100 test metrics on ml-100k", latex)
            self.assertEqual(latex.count("\\begin{table*}"), 4)
            selected = json.loads(
                paths["hyperparameters_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                selected["ml-1m"]["1.0"]["ease"],
                {"regularization": 500.0},
            )

    def test_accepts_summary_path_and_writes_resource_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "summary.json"
            source.write_text(json.dumps(synthetic_summary()), encoding="utf-8")
            paths = write_reports(source, root / "reports")
            with paths["resources_csv"].open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            names = {row["resource"] for row in rows}
            self.assertTrue(
                {
                    "training_seconds",
                    "inference_seconds",
                    "cpu_process_peak_mib",
                    "positive_catalog_items",
                    "validation_target_positive_support_rate",
                    "test_target_positive_support_rate",
                }.issubset(names)
            )
            self.assertEqual(
                next(
                    row
                    for row in rows
                    if row["resource"] == "training_seconds"
                )["unit"],
                "s",
            )
            self.assertEqual(
                next(
                    row
                    for row in rows
                    if row["resource"] == "cpu_process_peak_mib"
                )["unit"],
                "MiB",
            )
            self.assertEqual(
                next(
                    row
                    for row in rows
                    if row["resource"]
                    == "validation_target_positive_support_rate"
                )["unit"],
                "fraction",
            )

    def test_search_audit_groups_validation_by_embedding(self):
        records = [
            {
                "dataset": "ml-100k",
                "proportion": 1.0,
                "model": "mf-mse",
                "objective": "hit@10",
                "config_index": 0,
                "parameters": {"embedding_dim": 64, "learning_rate": 0.001},
                "validation_scores": [0.4, 0.6],
                "validation_seed_values": [11, 12],
                "mean_validation_score": 0.5,
            },
            {
                "dataset": "ml-100k",
                "proportion": 1.0,
                "model": "mf-mse",
                "objective": "hit@10",
                "config_index": 1,
                "parameters": {"embedding_dim": 128, "learning_rate": 0.001},
                "validation_scores": [0.7, 0.9],
                "validation_seed_values": [11, 12],
                "mean_validation_score": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_search_audit(records, directory)
            with paths["embedding_validation_summary_csv"].open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["embedding_dim"] for row in rows], ["64", "128"])
            self.assertEqual(rows[1]["best_config_index"], "1")
            with paths["search_configurations_csv"].open(
                newline="", encoding="utf-8"
            ) as handle:
                configurations = list(csv.DictReader(handle))
            self.assertEqual(
                configurations[0]["validation_seed_values_json"], "[11,12]"
            )
            self.assertAlmostEqual(
                float(rows[1]["best_validation_score"]), 0.8
            )

    def test_invalid_statistics_are_rejected(self):
        summary = synthetic_summary()
        summary["ml-100k"]["1.0"]["mf-mse"]["metrics"]["hit@10"]["n"] = 0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "n must be positive"):
                write_reports(summary, Path(directory))


if __name__ == "__main__":
    unittest.main()
