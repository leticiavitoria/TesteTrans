"""Pede ao Claude que sugira pontos de quebra de cena (em segundos)."""

from __future__ import annotations

import json
import os
import re
import sys

from .transcribe import Word


SYSTEM_PROMPT = (
    "Voce recebe a transcricao de um audio com timestamps por palavra. "
    "Sua tarefa e identificar bons momentos para encerrar uma CENA narrativa. "
    "Bons cortes acontecem em: mudanca de topico, fim de ideia clara, "
    "introducao de novo capitulo/item de lista, pergunta-retorica seguida de resposta, "
    "pausa longa, virada emocional. Evite cortar no meio de uma frase. "
    "Use o conteudo do texto como guia principal — uma cena pode ser curta (poucos segundos) "
    "ou longa (varios minutos). Nao force cenas de tamanho similar. "
    "Responda APENAS com JSON valido no formato: "
    '{\"cut_points_seconds\": [12.4, 31.7, ...]} '
    "onde cada numero e o instante (em segundos) onde a cena DEVE terminar."
)


def _log(msg: str) -> None:
    print(f"[scene_planner] {msg}", file=sys.stderr, flush=True)


def suggest_cut_points(words: list[Word], language: str = "auto") -> list[float]:
    """Retorna lista de segundos sugeridos para encerrar cenas.

    Em caso de falha retorna lista vazia, mas registra o motivo no stderr
    para que o usuario saiba por que nao houve corte inteligente.
    """
    if not words:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log("ANTHROPIC_API_KEY nao definida — pulando Claude. "
             "Defina a variavel de ambiente para ativar a divisao inteligente de cenas.")
        return []

    try:
        from anthropic import Anthropic
    except ImportError:
        _log("Pacote 'anthropic' nao instalado — rode: pip install anthropic")
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
            max_tokens=4096,
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
    except Exception as exc:
        _log(f"Erro chamando a API do Claude: {exc}")
        return []

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        _log("Resposta do Claude nao contem JSON — usando heuristica.")
        return []
    try:
        data = json.loads(match.group(0))
        cuts = sorted(float(c) for c in data.get("cut_points_seconds", []) if isinstance(c, (int, float)))
        _log(f"Claude sugeriu {len(cuts)} pontos de corte.")
        return cuts
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _log(f"JSON invalido do Claude ({exc}) — usando heuristica.")
        return []
