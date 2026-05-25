import unittest

import numpy as np
import torch

from src.models.edaic_feature_regression_baseline import EDAICFeatureRegressionBaseline
from src.utils.metrics import regression_metrics


class RegressionMetricsTest(unittest.TestCase):
    def test_computes_mae_rmse_and_ccc(self):
        targets = np.array([0.0, 2.0, 4.0])
        predictions = np.array([1.0, 2.0, 5.0])

        metrics = regression_metrics(targets, predictions)

        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(metrics["ccc"], 0.8888888888888888)

    def test_returns_zero_ccc_for_constant_targets_and_predictions(self):
        targets = np.array([4.0, 4.0, 4.0])
        predictions = np.array([4.0, 4.0, 4.0])

        metrics = regression_metrics(targets, predictions)

        self.assertEqual(metrics["ccc"], 0.0)


class EDAICFeatureRegressionBaselineTest(unittest.TestCase):
    def test_outputs_one_score_per_item_for_each_modality(self):
        text_embeddings = torch.ones(3, 4)
        audio_embeddings = torch.ones(3, 5)

        for modality in ("text", "audio", "both"):
            with self.subTest(modality=modality):
                model = EDAICFeatureRegressionBaseline(
                    text_dim=4,
                    audio_dim=5,
                    hidden_dim=6,
                    dropout=0.0,
                    modality=modality,
                )

                predictions = model(text_embeddings=text_embeddings, audio_embeddings=audio_embeddings)

                self.assertEqual(predictions.shape, torch.Size([3]))


if __name__ == "__main__":
    unittest.main()
