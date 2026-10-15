"""Índice de acceso aleatorio sobre un JSONL de tracking: ``frame_idx`` ->
offset de byte de su línea. Permite servir una ventana de tiempo de un
partido completo (250-330 MB por fichero) leyendo solo las líneas necesarias.

El índice son dos arrays de int64 (frames y offsets) en un fichero binario
``<nombre>.idx``; construirlo es una pasada secuencial que no parsea JSON
(solo busca ``"frame_idx": N`` al principio de cada línea).
"""

from __future__ import annotations

import json
import re
from array import array
from bisect import bisect_left, bisect_right
from pathlib import Path

_FRAME_RE = re.compile(rb'"frame_idx":\s*(\d+)')


class TrackingIndex:
    def __init__(self, frames: array, offsets: array) -> None:
        self.frames = frames
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.frames)

    @classmethod
    def build(cls, jsonl_path: Path) -> "TrackingIndex":
        frames, offsets = array("q"), array("q")
        with open(jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    m = _FRAME_RE.search(line, 0, 64) or _FRAME_RE.search(line)
                    if m is None:
                        raise ValueError(f"{jsonl_path}: line without frame_idx at byte {offset}")
                    frames.append(int(m.group(1)))
                    offsets.append(offset)
                offset += len(line)
        return cls(frames, offsets)

    def save(self, idx_path: Path) -> None:
        with open(idx_path, "wb") as f:
            f.write(len(self.frames).to_bytes(8, "little"))
            self.frames.tofile(f)
            self.offsets.tofile(f)

    @classmethod
    def load(cls, idx_path: Path) -> "TrackingIndex":
        with open(idx_path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            frames, offsets = array("q"), array("q")
            frames.fromfile(f, n)
            offsets.fromfile(f, n)
        return cls(frames, offsets)

    def read_window(self, jsonl_path: Path, from_frame: int, to_frame: int) -> list[dict]:
        """Registros con ``from_frame <= frame_idx <= to_frame``, en orden."""
        lo = bisect_left(self.frames, from_frame)
        hi = bisect_right(self.frames, to_frame)
        if lo >= hi:
            return []
        with open(jsonl_path, "rb") as f:
            f.seek(self.offsets[lo])
            records = []
            for _ in range(hi - lo):
                line = f.readline()
                if line.strip():
                    records.append(json.loads(line))
        return records
