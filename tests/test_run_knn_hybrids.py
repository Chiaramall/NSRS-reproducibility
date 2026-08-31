import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_knn_hybrids.py"
)
SPEC = importlib.util.spec_from_file_location("run_knn_hybrids", MODULE_PATH)
knn = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = knn
SPEC.loader.exec_module(knn)


class ItemKNNHybridTest(unittest.TestCase):
    def setUp(self):
        self.train = sp.csr_matrix(
            np.asarray(
                [
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [0, 0, 1, 1],
                ],
                dtype=np.float32,
            )
        )
        self.content = sp.csr_matrix(
            np.asarray(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                ],
                dtype=np.float32,
            )
        )

    def test_profile_dimensions_and_hybrid_weight(self):
        cf = knn.build_item_profiles(
            "ItemKNN-CF", self.train, None, "none", "none", 1.0
        )
        cbf = knn.build_item_profiles(
            "ItemKNN-CBF", self.train, self.content, "none", "none", 1.0
        )
        hybrid = knn.build_item_profiles(
            "ItemKNN-CFCBF",
            self.train,
            self.content,
            "none",
            "none",
            2.0,
        )
        self.assertEqual(cf.shape, (4, 3))
        self.assertEqual(cbf.shape, (4, 3))
        self.assertEqual(hybrid.shape, (4, 6))
        np.testing.assert_array_equal(hybrid[:, :3].toarray(), cf.toarray())
        np.testing.assert_array_equal(
            hybrid[:, 3:].toarray(), 2.0 * cbf.toarray()
        )

    def test_similarity_is_topk_without_diagonal(self):
        similarity = knn.topk_cosine_similarity(
            self.content,
            neighbors=1,
            shrink=0.0,
            normalize=True,
            block_size=2,
        )
        self.assertEqual(similarity.shape, (4, 4))
        self.assertTrue(np.all(similarity.getnnz(axis=1) <= 1))
        np.testing.assert_array_equal(similarity.diagonal(), np.zeros(4))
        self.assertAlmostEqual(float(similarity[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(similarity[2, 3]), 1.0, places=6)

    def test_zero_content_weight_recovers_cf_similarity(self):
        cf = knn.build_item_profiles(
            "ItemKNN-CF", self.train, None, "none", "none", 1.0
        )
        hybrid = knn.build_item_profiles(
            "ItemKNN-CFCBF",
            self.train,
            self.content,
            "none",
            "none",
            0.0,
        )
        cf_similarity = knn.topk_cosine_similarity(
            cf, neighbors=3, shrink=2.0, normalize=True, block_size=2
        )
        hybrid_similarity = knn.topk_cosine_similarity(
            hybrid, neighbors=3, shrink=2.0, normalize=True, block_size=2
        )
        np.testing.assert_allclose(
            cf_similarity.toarray(), hybrid_similarity.toarray()
        )

    def test_evaluation_masks_seen_items_and_matches_metrics(self):
        train = sp.csr_matrix(
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        similarity = sp.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.9, 0.8],
                    [0.7, 0.0, 0.6],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        metrics = knn.evaluate(
            train,
            similarity,
            [np.asarray([1]), np.asarray([0])],
            topk=1,
            batch_size=2,
            max_users=0,
        )
        for metric in knn.METRICS:
            self.assertAlmostEqual(metrics[metric], 100.0)


if __name__ == "__main__":
    unittest.main()
