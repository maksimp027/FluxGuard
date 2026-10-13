import os
import sys
import unittest
import numpy as np
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import generate_synthetic_data


class TestMLPipeline(unittest.TestCase):
    def test_synthetic_data_schema(self):
        X, y = generate_synthetic_data(n_samples=100)

        self.assertEqual(X.shape[1], 19, "Feature matrix should have exactly 19 columns.")
        self.assertEqual(len(y), 100, "Target array should match the number of samples.")

        unique_targets = np.unique(y)
        self.assertTrue(
            set(unique_targets).issubset({0, 1}),
            "Target must be binary (0 or 1).",
        )
        self.assertTrue(
            np.all(X[:, 18] == 0.0),
            "Index 18 (padding) must be 0.0 for all samples.",
        )

    def test_xgboost_output_shape(self):
        X = np.random.rand(10, 19)
        y = np.random.randint(0, 2, size=10)

        model = xgb.XGBClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X, y)

        probas = model.predict_proba(X)
        self.assertEqual(
            probas.shape,
            (10, 2),
            "XGBClassifier predict_proba should output shape (N, 2).",
        )


if __name__ == "__main__":
    unittest.main()
