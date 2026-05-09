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
    "CONCEITO: cada cena comeca com um prompt BASE e e seguida de extensoes (EXT) "
    "que detalham o MESMO mini-contexto. Quando o mini-contexto muda, uma nova cena "
    "(novo BASE) deve comecar — mesmo que ainda estejamos dentro do mesmo item da lista. "
    "Pense em cada cena como uma 'unidade visual' que ilustraria um momento especifico do narrativa.\n\n"
    "REGRAS OBRIGATORIAS para audios em formato lista TOP-N:\n"
    "1. A INTRODUCAO (antes do primeiro item da lista) e SEMPRE pelo menos uma cena propria, "
    "podendo ser quebrada em varias se houver mini-contextos distintos (gancho, previa dos itens, CTA inicial). "
    "Encerre-a no instante imediatamente ANTES do narrador anunciar o primeiro item.\n"
    "2. CADA ITEM da lista (10, 9, 8, ... 1) deve iniciar uma NOVA cena. "
    "Coloque um cut_point no instante imediatamente ANTES do narrador anunciar o numero/titulo do proximo item. "
    "Esta fronteira e OBRIGATORIA.\n"
    "3. DENTRO de cada item, ADICIONE mais cut_points sempre que o foco narrativo mudar de mini-contexto. "
    "Exemplos de mini-contextos que merecem cena propria: "
    "(a) descricao fisica/contexto historico do objeto, "
    "(b) descoberta/escavacao/quem encontrou, "
    "(c) analise cientifica/estudo especifico, "
    "(d) mistério/teorias concorrentes, "
    "(e) implicacao/conclusao do item. "
    "Itens longos devem virar 3 a 6 cenas. Itens curtos podem ter 1 a 2 cenas.\n"
    "4. O FECHAMENTO/OUTRO (recapitulacao, CTA, despedida apos o item 1) "
    "deve ser pelo menos uma cena propria, possivelmente quebrada em recapitulacao + CTA.\n\n"
    "REGRAS GERAIS:\n"
    "- Nao corte no meio de uma frase. Prefira o fim da sentenca antes do novo mini-contexto.\n"
    "- Cenas podem ter durações muito diferentes. NAO force tamanhos similares.\n"
    "- O numero TOTAL de cenas deve ser MAIOR que (intro + N itens + outro). "
    "Para um Top 10, espere algo como 25 a 50 cenas no total, nao apenas 12.\n"
    "- Quando estiver em duvida entre cortar ou nao cortar dentro de um item, PREFIRA cortar — "
    "cenas mais granulares sao melhores que cenas longas demais.\n\n"
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
