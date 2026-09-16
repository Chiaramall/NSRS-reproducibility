import unittest

import numpy as np

from movielens_paper_benchmark.metrics import evaluate_sampled_ranking


class SampledRankingMetricTest(unittest.TestCase):
    def test_metrics_and_candidate_order_tie_break_are_exact(self):
        scores = np.zeros((2, 6), dtype=np.float64)
        scores[0, 3] = 2.0
        scores[0, 4] = 1.0
        scores[1, 0] = 1.0
        scores[1, 1] = 1.0
        result = evaluate_sampled_ranking(
            lambda users: scores[np.asarray(users)],
            user_ids=[0, 1],
            targets={0: 4, 1: 1},
            candidate_items={0: [1, 3, 4], 1: [0, 1, 5]},
            n_items=6,
            ks=[1, 2],
            batch_size=2,
        )
        self.assertAlmostEqual(result.means["hit@1"], 0.0)
        self.assertAlmostEqual(result.means["hit@2"], 1.0)
        expected_ndcg = 1.0 / np.log2(3.0)
        self.assertAlmostEqual(result.means["ndcg@2"], expected_ndcg)

    def test_rejects_duplicate_candidates(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_sampled_ranking(
                lambda users: np.zeros((len(users), 3)),
                user_ids=[0],
                targets={0: 1},
                candidate_items={0: [1, 1, 2]},
                n_items=3,
            )


if __name__ == "__main__":
    unittest.main()
