"""Wrapper sobre faster-whisper para obter palavras com timestamps."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str


def transcribe(audio_path: str, model_size: str = "large-v3") -> tuple[list[Word], str]:
    """Transcreve o áudio e retorna (palavras, idioma_detectado).

    Cada Word traz start/end em segundos e o texto bruto.
    """
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "auto")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )

    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), text=text))

    words.sort(key=lambda x: x.start)
    return words, info.language
