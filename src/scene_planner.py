"""Pede ao Claude que sugira pontos de quebra de cena (em segundos)."""

from __future__ import annotations

import json
import os
import re
import sys

from .transcribe import Word


SYSTEM_PROMPT = (
    "Voce recebe a transcricao de um audio com timestamps por palavra. "
    "O audio costuma ser um video de lista TOP-N (Top 10, Top 5, etc). "
    "Sua tarefa e identificar os instantes onde cada CENA narrativa termina.\n\n"
    "REGRAS OBRIGATORIAS para audios em formato lista TOP-N:\n"
    "1. A INTRODUCAO (antes do primeiro item da lista) e SEMPRE uma cena propria. "
    "Encerre-a no instante imediatamente ANTES do narrador anunciar o primeiro item "
    "(ex.: \"10.\", \"Number 10\", \"In tenth place\", \"Decimo lugar\").\n"
    "2. CADA ITEM da lista (10, 9, 8, ... 1) deve iniciar uma NOVA cena. "
    "Coloque um cut_point no instante imediatamente ANTES do narrador anunciar o numero/titulo do proximo item. "
    "Exemplo: se o item 9 comeca aos 245.3s com \"9. Roman dodecahedra\", "
    "entao o cut_point que encerra a cena do item 10 deve ser ~245.3s "
    "(ou no fim da palavra anterior, para nao cortar no meio de frase).\n"
    "3. O FECHAMENTO/OUTRO (recapitulacao, call-to-action, despedida apos o item 1) "
    "deve ser uma cena propria.\n\n"
    "REGRAS GERAIS:\n"
    "- Nao corte no meio de uma frase. Prefira o fim da sentenca anterior ao titulo do novo item.\n"
    "- Cenas podem ter durações muito diferentes (alguns segundos ou varios minutos). NAO force tamanhos similares.\n"
    "- Se algum item for muito longo, voce PODE adicionar cortes internos em mudancas de subtopico, "
    "mas a prioridade absoluta e marcar a fronteira entre itens.\n\n"
    "Responda APENAS com JSON valido no formato: "
    '{\"cut_points_seconds\": [12.4, 31.7, ...]} '
    "onde cada numero e o instante (em segundos) onde uma cena DEVE terminar."
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
