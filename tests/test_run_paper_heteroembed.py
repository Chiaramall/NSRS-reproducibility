import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_paper_heteroembed.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_paper_heteroembed", MODULE_PATH
)
hetero = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hetero
SPEC.loader.exec_module(hetero)


class HeteroEmbedTest(unittest.TestCase):
    def test_subsampling_keeps_rare_words_more_often(self):
        frequency = np.asarray([10000.0, 100.0, 1.0])
        rates = hetero.subsampling_rates(frequency, 1e-4)
        self.assertLess(rates[0], rates[1])
        self.assertLessEqual(rates[1], rates[2])
        self.assertLessEqual(float(rates.max()), 1.0)

    def test_relation_tail_sampling_and_missing_weight(self):
        relation = hetero.RelationData(
            name="brand",
            tail_entity="brand",
            offsets=np.asarray([0, 2, 2, 3], dtype=np.int64),
            tails=np.asarray([4, 5, 7], dtype=np.int64),
            frequency=np.ones(8),
        )
        tails, weights = hetero.sample_relation_tails(
            relation,
            np.asarray([0, 1, 2], dtype=np.int64),
            np.random.default_rng(123),
        )
        self.assertIn(int(tails[0]), (4, 5))
        self.assertEqual(int(tails[1]), 0)
        self.assertEqual(int(tails[2]), 7)
        np.testing.assert_array_equal(weights, [1.0, 0.0, 1.0])

    def test_original_nce_objective_is_finite_and_differentiable(self):
        sizes = {
            "user": 3,
            "product": 4,
            "word": 5,
            "related_product": 4,
            "brand": 2,
            "categories": 3,
        }
        torch.manual_seed(123)
        model = hetero.HeteroEmbed(sizes, 8, ("brand",))
        heads = torch.as_tensor([0, 1, 2])
        tails = torch.as_tensor([0, 1, 2])
        weights = torch.ones(3)
        distribution = torch.ones(4) / 4
        loss = model.relation_loss(
            "user",
            "product",
            "purchase",
            heads,
            tails,
            weights,
            distribution,
            negative_samples=2,
            l2_lambda=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.entities["user"].weight.grad)
        self.assertIsNotNone(model.relation_vectors["purchase"].grad)

    def test_recommendation_score_matches_translation_equation(self):
        sizes = {
            "user": 2,
            "product": 3,
            "word": 2,
            "related_product": 3,
            "brand": 1,
            "categories": 1,
        }
        model = hetero.HeteroEmbed(sizes, 2, ())
        with torch.no_grad():
            model.entities["user"].weight[:] = torch.tensor(
                [[1.0, 2.0], [0.0, 1.0]]
            )
            model.entities["product"].weight[:] = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            )
            model.relation_vectors["purchase"][:] = torch.tensor([0.5, -0.5])
            model.tail_bias["purchase"][:] = torch.tensor([0.1, 0.2, 0.3])
        scores = model.recommendation_scores(
            torch.as_tensor([0]), use_bias=True
        )
        expected = torch.tensor([[1.6, 1.7, 3.3]])
        torch.testing.assert_close(scores, expected)


if __name__ == "__main__":
    unittest.main()
