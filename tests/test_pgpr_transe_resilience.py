import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "models"
    / "pgpr"
    / "data_utils.py"
)
assert MODULE_PATH.exists(), "The integrated PGPR source is missing"
SPEC = importlib.util.spec_from_file_location("pgpr_data_utils", MODULE_PATH)
pgpr_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pgpr_data)


class PGPRTransEResilienceTest(unittest.TestCase):
    def test_fully_subsampled_terminal_batch_remains_two_dimensional(self):
        relation = SimpleNamespace(data=[[]])
        dataset = SimpleNamespace(
            review=SimpleNamespace(size=1, data=[(0, 0, [0])]),
            word_sampling_rate=np.asarray([0.0]),
            produced_by=relation,
            belongs_to=relation,
            also_bought=relation,
            also_viewed=relation,
            bought_together=relation,
        )
        loader = pgpr_data.AmazonDataLoader(dataset, batch_size=64)

        batch = loader.get_batch()

        self.assertEqual(batch.shape, (0, 8))
        self.assertEqual(batch.dtype, np.int64)
        self.assertFalse(loader.has_next())


if __name__ == "__main__":
    unittest.main()
