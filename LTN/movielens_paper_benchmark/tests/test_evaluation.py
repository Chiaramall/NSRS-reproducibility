import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from movielens_paper_benchmark.data import (
    Feedback,
    Interaction,
    MovieLensData,
)
from movielens_paper_benchmark.evaluation import (
    BenchmarkRunner,
    build_sampled_candidates,
    evaluation_seen_items,
)
from movielens_paper_benchmark.splits import temporal_leave_two_out


def tiny_data():
    events = (
        Interaction(0, 0, 5.0, 10, Feedback.POSITIVE),
        Interaction(0, 1, 1.0, 20, Feedback.NEGATIVE),
        Interaction(0, 2, 4.0, 30, Feedback.POSITIVE),
        Interaction(0, 3, 2.0, 40, Feedback.NEGATIVE),
        Interaction(0, 4, 5.0, 50, Feedback.POSITIVE),
        Interaction(0, 5, 3.0, 60, Feedback.NEUTRAL),
    )
    return MovieLensData(
        variant="ml-100k",
        interactions=events,
        raw_user_ids=(1,),
        raw_item_ids=tuple(range(1, 111)),
        genre_names=("A", "B"),
        item_genres=tuple((item % 2,) for item in range(110)),
        item_titles=tuple(str(item) for item in range(110)),
    )


def tiny_config(output_root):
    return {
        "data_root": "unused",
        "output_root": str(output_root),
        "datasets": ["ml-100k"],
        "models": ["ease"],
        "proportions": [1.0],
        "positive_threshold": 4.0,
        "negative_threshold": 2.0,
        "min_train_positives": 1,
        "final_seeds": 1,
        "checkpoint_policy": "none",
        "search": {
            "strategy": "random",
            "budget": 2,
            "validation_seeds": 2,
            "seed": 9,
            "objective": "hit@10",
        },
        "training": {
            "max_epochs": 1,
            "patience": 1,
            "negative_samples": 0,
            "genre_samples_per_user": 1,
        },
        "ranking": {
            "protocol": "sampled-100",
            "negative_candidates": 100,
            "candidate_seed": 2026,
            "ks": [10],
            "user_batch_size": 8,
        },
        "spaces": {"ease": {"regularization": [1.0, 2.0]}},
    }


class ChronologicalMaskTest(unittest.TestCase):
    def test_test_mask_contains_intervening_events_but_not_future(self):
        data = tiny_data()
        split = temporal_leave_two_out(data)
        self.assertEqual(evaluation_seen_items(data, split, "validation"), {0: {0, 1}})
        self.assertEqual(
            evaluation_seen_items(data, split, "test"),
            {0: {0, 1, 2, 3}},
        )


class SampledCandidateTest(unittest.TestCase):
    def test_candidates_are_fixed_unique_and_never_use_rated_negatives(self):
        data = tiny_data()
        split = temporal_leave_two_out(data)
        first = build_sampled_candidates(data, split, "validation", 100, 2026)
        repeated = build_sampled_candidates(data, split, "validation", 100, 2026)
        changed = build_sampled_candidates(data, split, "validation", 100, 2027)
        np = __import__("numpy")
        np.testing.assert_array_equal(first[0], repeated[0])
        self.assertFalse(np.array_equal(first[0], changed[0]))
        self.assertEqual(len(first[0]), 101)
        self.assertEqual(len(set(first[0].tolist())), 101)
        self.assertEqual(first[0][-1], split.validation[0].item_idx)
        rated = {event.item_idx for event in data.interactions}
        self.assertTrue(set(first[0][:-1]).isdisjoint(rated))


class SearchResumeTest(unittest.TestCase):
    def test_resume_inside_configuration_and_never_touch_test(self):
        data = tiny_data()
        split = temporal_leave_two_out(data)
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(tiny_config(directory), verbose=False)
            runner.load_dataset = mock.Mock(return_value=(data, split))
            training_payload = {
                "training_seconds": 0.01,
                "memory": {},
                "diagnostics": {"epochs_trained": 1},
            }

            with mock.patch(
                "movielens_paper_benchmark.evaluation.train_model",
                return_value=(object(), training_payload),
            ), mock.patch(
                "movielens_paper_benchmark.evaluation.evaluate_model",
                side_effect=[
                    {"hit@10": 0.1},
                    RuntimeError("simulated interruption"),
                ],
            ) as evaluate:
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    runner.search(["ml-100k"], ["ease"])
                self.assertTrue(all(call.args[3] == "validation" for call in evaluate.call_args_list))

            state_path = Path(directory) / "search/ml-100k/1.0/ease.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["completed_fits"], 1)
            self.assertEqual(state["completed"], 0)
            self.assertEqual(
                len(state["records"][0]["validation_seed_values"]), 1
            )
            self.assertFalse(state["test_metrics_computed"])

            with mock.patch(
                "movielens_paper_benchmark.evaluation.train_model",
                return_value=(object(), training_payload),
            ) as train, mock.patch(
                "movielens_paper_benchmark.evaluation.evaluate_model",
                return_value={"hit@10": 0.5},
            ) as evaluate:
                runner.search(["ml-100k"], ["ease"])
                self.assertEqual(train.call_count, 3)
                self.assertTrue(all(call.args[3] == "validation" for call in evaluate.call_args_list))

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["completed_fits"], 4)
            self.assertEqual(state["completed"], 2)
            self.assertEqual(state["planned"], 2)
            self.assertEqual(len(state["records"][0]["validation_scores"]), 2)
            self.assertEqual(
                len(state["records"][0]["validation_seed_values"]), 2
            )
            self.assertIsNotNone(state["best_parameters"])
            self.assertFalse(state["test_metrics_computed"])

            records = runner.search_records(["ml-100k"], ["ease"])
            self.assertEqual(
                records[0]["validation_seed_values"],
                state["records"][0]["validation_seed_values"],
            )
            state["objective"] = "ndcg@10"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed fields"):
                runner.search_records(["ml-100k"], ["ease"])


if __name__ == "__main__":
    unittest.main()
