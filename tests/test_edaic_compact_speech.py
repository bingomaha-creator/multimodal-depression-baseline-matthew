import unittest

from src.utils.compact_speech import (
    TranscriptPiece,
    build_compact_audio_chunks,
    build_compact_speech_chunks,
    filter_transcript_pieces,
)


class CompactSpeechTest(unittest.TestCase):
    def test_filters_bad_timestamps_and_sorts_valid_rows(self):
        pieces = [
            TranscriptPiece(620.2, 620.7, "bye-bye"),
            TranscriptPiece(14.3, 648.3, "bad long row"),
            TranscriptPiece(640.3, 643.7, "a real life person"),
            TranscriptPiece(10.0, 9.0, "bad reverse row"),
            TranscriptPiece(1.0, 1.5, ""),
        ]

        valid, counts = filter_transcript_pieces(
            pieces,
            max_raw_segment_seconds=30.0,
            audio_duration_seconds=700.0,
        )

        self.assertEqual([piece.text for piece in valid], ["bye-bye", "a real life person"])
        self.assertEqual(counts["too_long"], 1)
        self.assertEqual(counts["invalid_time"], 1)
        self.assertEqual(counts["empty_text"], 1)

    def test_splits_when_audio_or_text_threshold_would_be_exceeded(self):
        pieces = [
            TranscriptPiece(0.0, 4.0, "one two"),
            TranscriptPiece(5.0, 9.0, "three four"),
            TranscriptPiece(10.0, 16.0, "five six"),
            TranscriptPiece(17.0, 18.0, "seven eight nine ten"),
        ]

        chunks = build_compact_speech_chunks(
            pieces,
            token_counter=lambda text: len(text.split()),
            max_audio_chunk_seconds=8.0,
            max_text_chunk_tokens=5,
            max_chunks=10,
        )

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three four",
            "five six",
            "seven eight nine ten",
        ])
        self.assertEqual([round(chunk.audio_seconds, 1) for chunk in chunks], [8.0, 6.0, 1.0])

    def test_audio_only_chunks_ignore_text_length(self):
        pieces = [
            TranscriptPiece(0.0, 2.0, "one two three four five six"),
            TranscriptPiece(3.0, 5.0, "seven eight nine ten eleven twelve"),
            TranscriptPiece(6.0, 8.0, "thirteen fourteen fifteen sixteen"),
        ]

        chunks = build_compact_audio_chunks(
            pieces,
            max_audio_chunk_seconds=4.0,
            max_chunks=10,
        )

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three four five six seven eight nine ten eleven twelve",
            "thirteen fourteen fifteen sixteen",
        ])
        self.assertEqual([round(chunk.audio_seconds, 1) for chunk in chunks], [4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
