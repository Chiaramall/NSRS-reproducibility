import unittest

import numpy as np

from movielens_paper_benchmark.data import Feedback, Interaction, MovieLensData
from movielens_paper_benchmark.evaluation import train_and_evaluate
from movielens_paper_benchmark.splits import temporal_leave_two_out


def model_data():
    rows = []
    histories = (
        ((0, 5.0), (3, 1.0), (6, 4.0), (9, 2.0), (10, 5.0)),
        ((1, 5.0), (4, 1.0), (7, 4.0), (9, 2.0), (11, 5.0)),
        ((2, 5.0), (5, 1.0), (8, 4.0), (10, 2.0), (6, 5.0)),
    )
    for user, history in enumerate(histories):
        for timestamp, (item, rating) in enumerate(history, start=1):
            feedback = (
                Feedback.POSITIVE
                if rating >= 4
                else Feedback.NEGATIVE
                if rating <= 2
                else Feedback.NEUTRAL
            )
            rows.append(
                Interaction(user, item, rating, timestamp * 10, feedback)
            )
    return MovieLensData(
        variant="ml-100k",
        interactions=tuple(rows),
        raw_user_ids=(1, 2, 3),
        raw_item_ids=tuple(range(1, 113)),
        genre_names=("disliked", "liked"),
        item_genres=tuple((1,) if item in {0, 1, 2, 6, 7, 8, 10, 11} else (0,) for item in range(112)),
        item_titles=tuple(str(item) for item in range(112)),
    )


CONFIG = {
    "search": {"objective": "hit@10"},
    "training": {
        "max_epochs": 1,
        "patience": 1,
        "negative_samples": 1,
        "genre_samples_per_user": 2,
    },
    "ranking": {
        "protocol": "sampled-100",
        "negative_candidates": 100,
        "candidate_seed": 2026,
        "ks": [10],
        "user_batch_size": 2,
    },
}


PARAMETERS = {
    "mf-mse": {
        "embedding_dim": 4,
        "learning_rate": 0.01,
        "l2": 0.0,
        "biased": True,
        "batch_size": 3,
    },
    "bpr-mf": {
        "embedding_dim": 4,
        "learning_rate": 0.01,
        "l2": 0.0,
        "biased": False,
        "batch_size": 3,
    },
    "ease": {"regularization": 1.0},
    "lightgcn": {
        "embedding_dim": 4,
        "num_layers": 1,
        "learning_rate": 0.01,
        "l2": 0.0001,
        "batch_size": 3,
    },
    "ltn": {
        "embedding_dim": 4,
        "learning_rate": 0.01,
        "l2": 0.0,
        "biased": True,
        "batch_size": 3,
        "alpha": 0.05,
    },
    "ltn-genres": {
        "embedding_dim": 4,
        "learning_rate": 0.01,
        "l2": 0.0,
        "biased": True,
        "batch_size": 3,
        "alpha": 0.05,
        "p": 2,
        "genre_rule_weight": 0.5,
        "min_genre_support": 1,
    },
}


class AllModelsPipelineTest(unittest.TestCase):
    def test_every_model_trains_and_sampled_ranking_is_finite(self):
        data = model_data()
        split = temporal_leave_two_out(data)
        for name, parameters in PARAMETERS.items():
            with self.subTest(model=name):
                _, result = train_and_evaluate(
                    name,
                    parameters,
                    data,
                    split,
                    split,
                    seed=17,
                    config=CONFIG,
                    device="cpu",
                )
                expected_epochs = 0 if name == "ease" else 1
                self.assertEqual(result["diagnostics"]["epochs_trained"], expected_epochs)
                self.assertGreater(result["training_seconds"], 0.0)
                target_support = result["diagnostics"]["target_support"]
                self.assertEqual(target_support["evaluation_users"], 3)
                self.assertLessEqual(
                    target_support["validation_targets_with_positive_support"],
                    target_support["evaluation_users"],
                )
                self.assertLessEqual(
                    target_support["test_targets_with_positive_support"],
                    target_support["evaluation_users"],
                )
                for metric in ("hit@10", "ndcg@10"):
                    self.assertTrue(np.isfinite(result["metrics"][metric]))
                if name == "ltn-genres":
                    self.assertGreater(
                        result["diagnostics"]["n_genre_constraints"], 0
                    )


if __name__ == "__main__":
    unittest.main()
