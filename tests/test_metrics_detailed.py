import unittest

import numpy as np

from src.utils.metrics import detailed_classification_metrics


class DetailedClassificationMetricsTest(unittest.TestCase):
    def test_reports_binary_weighted_metrics_and_confusion_matrix(self):
        labels = np.array([0, 0, 0, 1])
        predictions = np.array([0, 0, 1, 1])

        metrics = detailed_classification_metrics(labels, predictions)

        self.assertAlmostEqual(metrics["acc"], 0.75)
        self.assertAlmostEqual(metrics["binary_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.7666666666666667)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
