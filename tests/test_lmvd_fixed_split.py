import tempfile
import unittest
from pathlib import Path

from src.utils.lmvd_split import load_fixed_split_indices


class LMVDFixedSplitTest(unittest.TestCase):
    def write_split(self, directory: str, content: str) -> Path:
        path = Path(directory) / "lmvd_labels.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_maps_leading_zero_ids_to_cache_indices(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_path = self.write_split(
                tmp_dir,
                "index,label,fold\n001,1,train\n002,0,valid\n010,1,test\n",
            )
            items = [
                {"participant_id": "1", "label": 1},
                {"participant_id": "002", "label": 0},
                {"participant_id": "10", "label": 1},
            ]

            split = load_fixed_split_indices(split_path, items)

            self.assertEqual(split["train"].tolist(), [0])
            self.assertEqual(split["valid"].tolist(), [1])
            self.assertEqual(split["test"].tolist(), [2])

    def test_rejects_label_mismatch_between_split_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_path = self.write_split(
                tmp_dir,
                "index,label,fold\n001,0,train\n002,0,valid\n003,1,test\n",
            )
            items = [
                {"participant_id": "001", "label": 1},
                {"participant_id": "002", "label": 0},
                {"participant_id": "003", "label": 1},
            ]

            with self.assertRaisesRegex(ValueError, "Label mismatch.*001"):
                load_fixed_split_indices(split_path, items)

    def test_rejects_missing_cache_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_path = self.write_split(
                tmp_dir,
                "index,label,fold\n001,1,train\n002,0,valid\n",
            )
            items = [
                {"participant_id": "001", "label": 1},
                {"participant_id": "002", "label": 0},
                {"participant_id": "003", "label": 1},
            ]

            with self.assertRaisesRegex(ValueError, "missing.*003"):
                load_fixed_split_indices(split_path, items)

if __name__ == "__main__":
    unittest.main()
