from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TranscriptPiece:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CompactSpeechChunk:
    pieces: Tuple[TranscriptPiece, ...]

    @property
    def text(self) -> str:
        return " ".join(piece.text.strip() for piece in self.pieces if piece.text.strip())

    @property
    def audio_seconds(self) -> float:
        return sum(piece.duration for piece in self.pieces)


def filter_transcript_pieces(
    pieces: Iterable[TranscriptPiece],
    max_raw_segment_seconds: float,
    audio_duration_seconds: Optional[float] = None,
) -> Tuple[List[TranscriptPiece], Dict[str, int]]:
    counts = {
        "empty_text": 0,
        "invalid_time": 0,
        "too_long": 0,
        "out_of_bounds": 0,
    }
    valid: List[TranscriptPiece] = []
    for piece in pieces:
        text = piece.text.strip()
        if not text:
            counts["empty_text"] += 1
            continue
        if piece.start < 0 or piece.end <= piece.start:
            counts["invalid_time"] += 1
            continue
        if piece.duration > max_raw_segment_seconds:
            counts["too_long"] += 1
            continue
        if audio_duration_seconds is not None and (
            piece.start >= audio_duration_seconds or piece.end > audio_duration_seconds
        ):
            counts["out_of_bounds"] += 1
            continue
        valid.append(TranscriptPiece(start=piece.start, end=piece.end, text=text))

    valid.sort(key=lambda item: (item.start, item.end))
    return valid, counts


def build_compact_speech_chunks(
    pieces: Iterable[TranscriptPiece],
    token_counter: Callable[[str], int],
    max_audio_chunk_seconds: float,
    max_text_chunk_tokens: int,
    max_chunks: int,
) -> List[CompactSpeechChunk]:
    if max_audio_chunk_seconds <= 0:
        raise ValueError("max_audio_chunk_seconds must be positive")
    if max_text_chunk_tokens <= 0:
        raise ValueError("max_text_chunk_tokens must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    chunks: List[CompactSpeechChunk] = []
    current: List[TranscriptPiece] = []

    def flush_current() -> None:
        if current and len(chunks) < max_chunks:
            chunks.append(CompactSpeechChunk(pieces=tuple(current)))

    for piece in pieces:
        candidate = current + [piece]
        candidate_text = " ".join(item.text for item in candidate)
        candidate_audio_seconds = sum(item.duration for item in candidate)
        candidate_tokens = token_counter(candidate_text)
        should_split = bool(current) and (
            candidate_audio_seconds > max_audio_chunk_seconds
            or candidate_tokens > max_text_chunk_tokens
        )
        if should_split:
            flush_current()
            current = [piece]
            if len(chunks) >= max_chunks:
                break
        else:
            current = candidate

    if len(chunks) < max_chunks:
        flush_current()
    return chunks[:max_chunks]
