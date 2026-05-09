"""Wrapper sobre faster-whisper para obter palavras com timestamps."""

from __future__ import annotations

import os
import sys
import time
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

    print(
        f"Carregando modelo {model_size} (device={device}, compute_type={compute_type})...",
        file=sys.stderr,
        flush=True,
    )
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if device != "cpu" and ("cuda" in msg or "cudnn" in msg or "gpu" in msg):
            print(
                f"Aviso: GPU indisponível ({exc}). Usando CPU.",
                file=sys.stderr,
                flush=True,
            )
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            raise

    print("Iniciando transcricao (em CPU pode demorar varios minutos por minuto de audio)...",
          file=sys.stderr, flush=True)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )

    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    if total_duration:
        print(
            f"Idioma detectado: {info.language}. Duracao do audio: {total_duration:.1f}s.",
            file=sys.stderr,
            flush=True,
        )

    words: list[Word] = []
    started = time.monotonic()
    last_log = started
    for seg in segments:
        now = time.monotonic()
        if now - last_log >= 5.0:
            elapsed = now - started
            if total_duration > 0:
                pct = min(100.0, 100.0 * seg.end / total_duration)
                print(
                    f"  ...progresso: {seg.end:.1f}s/{total_duration:.1f}s "
                    f"({pct:.1f}%) em {elapsed:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"  ...progresso: {seg.end:.1f}s transcritos em {elapsed:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            last_log = now

        if not seg.words:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), text=text))

    print(f"Transcricao concluida em {time.monotonic() - started:.0f}s.",
          file=sys.stderr, flush=True)

    words.sort(key=lambda x: x.start)
    return words, info.language
