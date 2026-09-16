import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from movielens_paper_benchmark.config import load_config, validate_config
from movielens_paper_benchmark.data import Feedback, Interaction, MovieLensData
from movielens_paper_benchmark.evaluation import (
    BenchmarkRunner,
    _final_run_fingerprint,
    _memory_snapshot,
    training_positive_items,
)
from movielens_paper_benchmark.models.ltn import LTNRecommender
from movielens_paper_benchmark.models.mf import MatrixFactorization
from movielens_paper_benchmark.run import (
    _write_final_reports,
    exclusive_output_lock,
    resolve_device,
)
from movielens_paper_benchmark.search import sample_configurations
from movielens_paper_benchmark.splits import (
    derive_user_genre_preferences,
    temporal_leave_two_out,
)
from movielens_paper_benchmark.tests.test_evaluation import tiny_config, tiny_data


FULL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "full.json"
)


class StrictConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(str(FULL_CONFIG))

    def test_rejects_unknown_hyperparameter(self):
        config = copy.deepcopy(self.config)
        config["spaces"]["mf-mse"]["learnng_rate"] = [0.1]
        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            validate_config(config)

    def test_rejects_nonfinite_fractional_and_duplicate_candidates(self):
        config = copy.deepcopy(self.config)
        config["spaces"]["mf-mse"]["learning_rate"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_config(config)

        config = copy.deepcopy(self.config)
        config["spaces"]["mf-mse"]["embedding_dim"] = [64.5]
        with self.assertRaisesRegex(TypeError, "integer"):
            validate_config(config)

        config = copy.deepcopy(self.config)
        config["spaces"]["mf-mse"]["embedding_dim"] = [64, 64]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_config(config)

    def test_rejects_boolean_integer_fields(self):
        config = copy.deepcopy(self.config)
        config["final_seeds"] = True
        with self.assertRaisesRegex(TypeError, "integer"):
            validate_config(config)


    def test_requires_exact_sampled_100_protocol(self):
        config = copy.deepcopy(self.config)
        config["ranking"]["protocol"] = "full-catalog"
        with self.assertRaisesRegex(ValueError, "sampled-100"):
            validate_config(config)

        config = copy.deepcopy(self.config)
        config["ranking"]["negative_candidates"] = 99
        with self.assertRaisesRegex(ValueError, "equal 100"):
            validate_config(config)


class SearchCoverageTest(unittest.TestCase):
    def test_embedding_strata_are_balanced_and_deterministic(self):
        space = {
            "embedding_dim": [5, 10, 20, 64, 128, 256],
            "learning_rate": [0.001, 0.01],
            "l2": [0.0, 0.001],
        }
        first = sample_configurations(space, 18, 91)
        second = sample_configurations(space, 18, 91)
        self.assertEqual(first, second)
        counts = {
            dimension: sum(
                row["embedding_dim"] == dimension for row in first
            )
            for dimension in space["embedding_dim"]
        }
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(
            len({json.dumps(row, sort_keys=True) for row in first}),
            len(first),
        )


class ProtocolIsolationTest(unittest.TestCase):
    def test_negative_exclusions_contain_training_positives_only(self):
        data = tiny_data()
        split = temporal_leave_two_out(data)
        exclusions = training_positive_items(split.train, data.n_users)
        self.assertEqual(exclusions, {0: {0}})
        self.assertNotIn(split.validation[0].item_idx, exclusions[0])
        self.assertNotIn(split.test[0].item_idx, exclusions[0])

    def test_neutral_ratings_do_not_create_genre_support(self):
        data = MovieLensData(
            variant="ml-100k",
            interactions=(
                Interaction(0, 0, 5.0, 1, Feedback.POSITIVE),
                Interaction(0, 1, 3.0, 2, Feedback.NEUTRAL),
            ),
            raw_user_ids=(1,),
            raw_item_ids=(1, 2),
            genre_names=("Drama",),
            item_genres=((0,), (0,)),
            item_titles=("a", "b"),
        )
        preferences = derive_user_genre_preferences(
            data,
            data.interactions,
            min_support=2,
            liked_threshold=4.0,
            disliked_threshold=2.0,
        )
        self.assertEqual(preferences, ())

    def test_mf_and_ltn_use_identical_seeded_backbone(self):
        mf = MatrixFactorization(
            3,
            5,
            embedding_dim=4,
            objective="mse",
            biased=True,
            seed=37,
        )
        ltn = LTNRecommender(
            3,
            5,
            embedding_dim=4,
            biased=True,
            seed=37,
        )
        mf_state = mf.model.state_dict()
        ltn_state = ltn.model.state_dict()
        self.assertEqual(set(mf_state), set(ltn_state))
        for key in mf_state:
            torch.testing.assert_close(mf_state[key], ltn_state[key])

    def test_genre_weight_one_uses_original_stable_satagg(self):
        model = LTNRecommender(
            2,
            3,
            embedding_dim=2,
            genre_rule_weight=1.0,
            seed=5,
        )
        data_satisfaction = torch.tensor(0.8)
        genre_satisfaction = torch.tensor(0.6)
        expected = 1.0 - model.sat_agg(
            data_satisfaction, genre_satisfaction
        )
        actual = model._knowledge_base_loss(
            data_satisfaction, genre_satisfaction
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_ltn_inference_checkpoint_round_trip(self):
        model = LTNRecommender(
            3,
            5,
            embedding_dim=4,
            biased=True,
            seed=41,
        )
        expected = model.score_users([0, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ltn.pth"
            model.save(path)
            payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                payload["checkpoint_purpose"], "inference_only"
            )
            self.assertNotIn("optimizer", payload)
            restored = LTNRecommender.load(path, device="cpu")
            np.testing.assert_allclose(
                restored.score_users([0, 2]), expected
            )

    def test_checkpoint_requirement_changes_run_fingerprint(self):
        search_state = {
            "protocol_digest": "abc",
            "best_parameters": {"embedding_dim": 64},
        }
        without = _final_run_fingerprint(search_state, 0, 123, False)
        with_checkpoint = _final_run_fingerprint(
            search_state, 0, 123, True
        )
        self.assertNotEqual(without, with_checkpoint)

    def test_cpu_memory_is_an_empty_mapping(self):
        self.assertEqual(_memory_snapshot("cpu"), {})


class RunnerSafetyTest(unittest.TestCase):
    def test_output_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_output_lock(root):
                self.assertEqual(
                    (root / "benchmark.pid").read_text(encoding="ascii").strip(),
                    str(__import__("os").getpid()),
                )
                with self.assertRaisesRegex(
                    RuntimeError, "another benchmark process"
                ):
                    with exclusive_output_lock(root):
                        pass
            self.assertFalse((root / "benchmark.pid").exists())

    def test_final_summary_is_complete_by_default(self):
        data = tiny_data()
        split = temporal_leave_two_out(data)
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(tiny_config(directory), verbose=False)
            runner.load_dataset = lambda name: (data, split)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                runner.summarize(["ml-100k"], ["ease"])
            partial = runner.summarize(
                ["ml-100k"], ["ease"], require_complete=False
            )
            self.assertEqual(partial, {"ml-100k": {"1.0": {}}})


class DeviceValidationTest(unittest.TestCase):
    def test_auto_resolves_to_a_canonical_device(self):
        resolved = resolve_device("auto")
        if torch.cuda.is_available():
            self.assertRegex(resolved, r"^cuda:\d+$")
        else:
            self.assertEqual(resolved, "cpu")

    def test_invalid_cuda_index_is_rejected(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        invalid = torch.cuda.device_count()
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_device(f"cuda:{invalid}")


class ReportArtifactSafetyTest(unittest.TestCase):
    @staticmethod
    def _fake_write_reports(summary, output_dir):
        del summary
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        markdown = output / "results.md"
        latex = output / "results.tex"
        markdown.write_text("results\n", encoding="utf-8")
        latex.write_text("results\n", encoding="utf-8")
        return {"markdown": markdown, "latex": latex}

    def test_complete_report_removes_stale_partial_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            warning = reports / "PARTIAL_RESULTS_WARNING.txt"
            warning.write_text("stale", encoding="utf-8")
            runner = mock.Mock()
            runner.output_root = root
            runner.config = {"proportions": [1.0]}
            runner.summarize.return_value = {
                "ml-100k": {"1.0": {"ease": {"complete": True}}}
            }
            runner.search_records.return_value = []
            with mock.patch(
                "movielens_paper_benchmark.run.write_reports",
                side_effect=self._fake_write_reports,
            ):
                _write_final_reports(
                    runner, ["ml-100k"], ["ease"], allow_partial=True
                )
            self.assertFalse(warning.exists())
            self.assertTrue((reports / "search_configurations.csv").exists())
            self.assertTrue(
                (reports / "embedding_validation_summary.csv").exists()
            )

    def test_partial_report_removes_stale_search_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            stale_paths = (
                reports / "search_configurations.csv",
                reports / "embedding_validation_summary.csv",
            )
            for path in stale_paths:
                path.write_text("stale", encoding="utf-8")
            runner = mock.Mock()
            runner.output_root = root
            runner.config = {"proportions": [1.0]}
            runner.summarize.return_value = {"ml-100k": {"1.0": {}}}
            runner.search_records.side_effect = RuntimeError("incomplete")
            with mock.patch(
                "movielens_paper_benchmark.run.write_reports",
                side_effect=self._fake_write_reports,
            ):
                _write_final_reports(
                    runner, ["ml-100k"], ["ease"], allow_partial=True
                )
            self.assertTrue(
                (reports / "PARTIAL_RESULTS_WARNING.txt").exists()
            )
            self.assertTrue(
                (reports / "results.md")
                .read_text(encoding="utf-8")
                .startswith("# WARNING: PARTIAL RESULTS")
            )
            self.assertTrue(all(not path.exists() for path in stale_paths))


if __name__ == "__main__":
    unittest.main()
