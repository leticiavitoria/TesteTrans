"""Pede ao Claude que sugira pontos de quebra de cena (em segundos)."""

from __future__ import annotations

import json
import os
import re

from .transcribe import Word


SYSTEM_PROMPT = (
    "Voce recebe a transcricao de um audio com timestamps por palavra. "
    "Sua tarefa e identificar bons momentos para encerrar uma CENA narrativa. "
    "Bons cortes acontecem em: mudanca de topico, fim de ideia, pergunta-resposta, "
    "pausa longa, virada emocional. Evite cortar no meio de uma frase. "
    "Responda APENAS com JSON valido no formato: "
    '{\"cut_points_seconds\": [12.4, 31.7, ...]} '
    "onde cada numero e o instante (em segundos) onde a cena DEVE terminar."
)


def suggest_cut_points(words: list[Word], language: str = "auto") -> list[float]:
    """Retorna lista de segundos sugeridos para encerrar cenas.

    Em caso de qualquer falha (sem API key, erro de rede, JSON invalido)
    retorna lista vazia: a heuristica deterministica seguira sozinha.
    """
    if not words:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    try:
        from anthropic import Anthropic
    except ImportError:
        return []

    transcript_lines = []
    for w in words:
        mm = int(w.start // 60)
        ss = w.start - mm * 60
        transcript_lines.append(f"[{mm:02d}:{ss:05.2f}] {w.text}")
    transcript = "\n".join(transcript_lines)

    total = words[-1].end
    user_msg = (
        f"Idioma detectado: {language}. Duracao total: {total:.1f}s.\n"
        f"Transcricao:\n{transcript}\n\n"
        "Devolva o JSON com os pontos de corte de cena."
    )

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
    except Exception:
        return []

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        cuts = data.get("cut_points_seconds", [])
        return sorted(float(c) for c in cuts if isinstance(c, (int, float)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
