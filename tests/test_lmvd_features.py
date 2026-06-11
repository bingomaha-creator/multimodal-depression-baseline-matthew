import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_lmvd_feature_cache import (
    build_feature_cache,
    discover_lmvd_samples,
    load_audio_embedding,
    load_video_embedding,
    read_lmvd_label,
    summarize_temporal_features,
)


class LMVDFeaturePreparationTest(unittest.TestCase):
    def test_discovers_samples_and_reads_label_from_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Video_feature").mkdir()
            (root / "Audio_feature").mkdir()
            (root / "label").mkdir()
            (root / "Video_feature" / "002.csv").write_text("frame, value\n1,0.1\n", encoding="utf-8")
            np.save(root / "Audio_feature" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "002_Depression.csv").write_text("1\n", encoding="utf-8")

            samples = discover_lmvd_samples(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].participant_id, "002")
            self.assertEqual(read_lmvd_label(samples[0].label_path), 1)

    def test_discovers_samples_recursively_inside_feature_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Video_feature" / "part_a").mkdir(parents=True)
            (root / "Audio_feature" / "part_a").mkdir(parents=True)
            (root / "label" / "part_a").mkdir(parents=True)
            (root / "Video_feature" / "part_a" / "002.csv").write_text("frame, value\n1,0.1\n", encoding="utf-8")
            np.save(root / "Audio_feature" / "part_a" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "part_a" / "002_Depression.csv").write_text("1\n", encoding="utf-8")

            samples = discover_lmvd_samples(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].participant_id, "002")

    def test_discovers_common_feature_directory_name_variants(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Visual_feature").mkdir()
            (root / "Acoustic_feature").mkdir()
            (root / "label").mkdir()
            (root / "Visual_feature" / "002.csv").write_text("frame, value\n1,0.1\n", encoding="utf-8")
            np.save(root / "Acoustic_feature" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "002_Depression.csv").write_text("1\n", encoding="utf-8")

            samples = discover_lmvd_samples(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].participant_id, "002")

    def test_discovers_combined_lmvd_feature_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "LMVD_Feature").mkdir()
            (root / "label").mkdir()
            (root / "LMVD_Feature" / "002.csv").write_text("frame, value\n1,0.1\n", encoding="utf-8")
            np.save(root / "LMVD_Feature" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "002_Depression.csv").write_text("1\n", encoding="utf-8")

            samples = discover_lmvd_samples(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].participant_id, "002")

    def test_empty_cache_error_reports_discovery_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Video_feature").mkdir()
            (root / "Audio_feature").mkdir()
            (root / "label").mkdir()
            (root / "Video_feature" / "002.csv").write_text("frame, value\n1,0.1\n", encoding="utf-8")
            np.save(root / "Audio_feature" / "003.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "004_Depression.csv").write_text("1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "video csv files=1.*audio npy files=1.*label csv files=1"):
                build_feature_cache(root, root / "lmvd_features.pt", save_torch=False)

    def test_loads_video_and_audio_embeddings_with_mean_std_pooling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video_path = root / "002.csv"
            audio_path = root / "002.npy"
            pd.DataFrame(
                {
                    "frame": [1, 2],
                    " timestamp": [0.0, 0.033],
                    " confidence": [0.9, 0.8],
                    " AU01_r": [1.0, 3.0],
                }
            ).to_csv(video_path, index=False)
            np.save(audio_path, np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32))

            video_embedding = load_video_embedding(video_path)
            audio_embedding = load_audio_embedding(audio_path)

            self.assertEqual(video_embedding.shape, (4,))
            self.assertTrue(np.allclose(video_embedding, np.array([0.85, 2.0, 0.05, 1.0], dtype=np.float32)))
            self.assertTrue(np.allclose(audio_embedding, np.array([2.0, 4.0, 1.0, 2.0], dtype=np.float32)))

    def test_builds_feature_cache_with_metadata_and_items(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Video_feature").mkdir()
            (root / "Audio_feature").mkdir()
            (root / "label").mkdir()
            pd.DataFrame({"frame": [1, 2], " AU01_r": [1.0, 3.0]}).to_csv(
                root / "Video_feature" / "002.csv",
                index=False,
            )
            np.save(root / "Audio_feature" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "002_Depression.csv").write_text("1\n", encoding="utf-8")
            output = root / "lmvd_features.pt"

            cache = build_feature_cache(root, output, save_torch=False)

            self.assertEqual(cache["metadata"]["video_dim"], 2)
            self.assertEqual(cache["metadata"]["audio_dim"], 6)
            self.assertEqual(cache["items"][0]["participant_id"], "002")
            self.assertEqual(cache["items"][0]["label"], 1)
            self.assertTrue(output.exists())

    def test_builds_feature_cache_with_aligned_video_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "LMVD_Feature").mkdir()
            (root / "label").mkdir()
            pd.DataFrame({"frame": [1, 2], " AU01_r": [1.0, 3.0]}).to_csv(
                root / "LMVD_Feature" / "001.csv",
                index=False,
            )
            pd.DataFrame({"frame": [1, 2], " AU02_r": [2.0, 4.0]}).to_csv(
                root / "LMVD_Feature" / "002.csv",
                index=False,
            )
            np.save(root / "LMVD_Feature" / "001.npy", np.ones((2, 3), dtype=np.float32))
            np.save(root / "LMVD_Feature" / "002.npy", np.ones((2, 3), dtype=np.float32))
            (root / "label" / "001_Depression.csv").write_text("0\n", encoding="utf-8")
            (root / "label" / "002_Depression.csv").write_text("1\n", encoding="utf-8")

            cache = build_feature_cache(root, root / "lmvd_features.pt", save_torch=False)

            self.assertEqual(cache["metadata"]["video_columns"], ["AU01_r", "AU02_r"])
            self.assertEqual(cache["metadata"]["video_dim"], 4)
            self.assertEqual(cache["items"][0]["video_embedding"].shape, cache["items"][1]["video_embedding"].shape)
            self.assertTrue(np.allclose(cache["items"][0]["video_embedding"], np.array([2.0, 0.0, 1.0, 0.0])))
            self.assertTrue(np.allclose(cache["items"][1]["video_embedding"], np.array([0.0, 3.0, 0.0, 1.0])))

    def test_summarize_temporal_features_rejects_empty_arrays(self):
        with self.assertRaises(ValueError):
            summarize_temporal_features(np.empty((0, 3), dtype=np.float32), source="empty.npy")


if __name__ == "__main__":
    unittest.main()
