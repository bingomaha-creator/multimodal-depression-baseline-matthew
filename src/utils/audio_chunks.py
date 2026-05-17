from __future__ import annotations

import math
from typing import List, Tuple


def select_uniform_chunk_ranges(
    total_length: int,
    chunk_length: int,
    max_chunks: int,
) -> List[Tuple[int, int]]:
    if total_length < 0:
        raise ValueError(f"total_length must be non-negative, got {total_length}")
    if chunk_length <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_length}")
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")

    if total_length <= chunk_length:
        return [(0, chunk_length)]

    needed_chunks = math.ceil(total_length / chunk_length)
    num_chunks = min(max_chunks, needed_chunks)
    max_start = total_length - chunk_length
    if num_chunks == 1:
        return [(0, chunk_length)]

    ranges: List[Tuple[int, int]] = []
    for index in range(num_chunks):
        start = round(index * max_start / (num_chunks - 1))
        ranges.append((start, start + chunk_length))
    return ranges
