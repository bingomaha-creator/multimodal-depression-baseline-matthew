import unittest

from src.utils.audio_chunks import select_uniform_chunk_ranges


class AudioChunkSelectionTest(unittest.TestCase):
    def test_selects_evenly_spaced_fixed_length_chunks(self):
        # A long waveform should be represented by chunks that cover early,
        # middle, and late parts instead of only the beginning.
        chunks = select_uniform_chunk_ranges(
            total_length=100,
            chunk_length=20,
            max_chunks=3,
        )

        self.assertEqual(chunks, [(0, 20), (40, 60), (80, 100)])

    def test_pads_short_waveform_to_one_chunk(self):
        # For short audio, the extractor will request one full chunk range.
        # The downstream feature extractor handles the actual zero padding.
        chunks = select_uniform_chunk_ranges(
            total_length=5,
            chunk_length=8,
            max_chunks=3,
        )

        self.assertEqual(chunks, [(0, 8)])


if __name__ == "__main__":
    unittest.main()
