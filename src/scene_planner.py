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


_ITEM_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "um": 1, "dois": 2, "tres": 3, "três": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "primeiro": 1, "primeira": 1, "segundo": 2, "segunda": 2,
    "terceiro": 3, "terceira": 3, "quarto": 4, "quarta": 4,
    "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6,
    "setimo": 7, "sétimo": 7, "setima": 7, "sétima": 7,
    "oitavo": 8, "oitava": 8, "nono": 9, "nona": 9,
    "decimo": 10, "décimo": 10, "decima": 10, "décima": 10,
}


def detect_item_boundaries(words: list[Word]) -> list[float]:
    """Detecta inicios de itens de lista TOP-N na transcricao.

    Estrategia: procurar uma SEQUENCIA DESCENDENTE de numeros (10 → 9 → 8 → ... → 1)
    em contexto de anuncio de item. Cada numero so e aceito se:
      - For exatamente o esperado (anterior - 1), OU
      - For o primeiro numero "alto" (>= 3) encontrado no audio
    e tiver contexto de titulo (palavra seguinte capitalizada / proper noun).

    Isso elimina falsos positivos como "10 tons", "1 ,000 years", "2 .8 billion",
    "3 ,100 degrees", "8 ,000 BCE", "11 ,600 years", "1994", etc.
    """
    n = len(words)
    candidates: list[tuple[int, float]] = []  # (numero, timestamp)

    for i, w in enumerate(words):
        token = w.text.strip().rstrip(".,;:!?\"'").lower()
        num: int | None = None
        boundary_idx = i  # palavra onde a cena nova comeca

        # Padrao "Number 10", "Numero 10" — fronteira na palavra "Number"
        if token in {"number", "numero", "número"} and i + 1 < n:
            nxt_token = words[i + 1].text.strip().rstrip(".,;:!?\"'").lower()
            m = re.match(r"^(\d{1,2})$", nxt_token)
            if m:
                cand = int(m.group(1))
                if 1 <= cand <= 20:
                    num = cand
                    boundary_idx = i  # cena nova comeca em "Number"

        # Padrao "10.", "9.", etc. — numero isolado, possivelmente com ponto
        if num is None:
            m = re.match(r"^(\d{1,2})\.?$", token)
            if m:
                cand = int(m.group(1))
                if 1 <= cand <= 20:
                    # Exige contexto de titulo: a proxima palavra deve comecar
                    # com letra maiuscula (proper noun) — descarta "10 tons", "10 to 20".
                    nxt_raw = words[i + 1].text.strip() if i + 1 < n else ""
                    nxt_first = nxt_raw[:1]
                    if nxt_first.isalpha() and nxt_first.isupper():
                        # Exige tambem que a palavra original termine com "." ou que
                        # exista uma pausa antes (>= 0.4s desde a palavra anterior),
                        # indicando que e fim de frase / anuncio, nao numero corrido.
                        ends_with_dot = w.text.strip().endswith(".")
                        prev_end = words[i - 1].end if i > 0 else w.start
                        gap = w.start - prev_end
                        if ends_with_dot or gap >= 0.4:
                            num = cand
                            boundary_idx = i

        # Padrao "decimo lugar", "tenth place"
        if num is None and token in _ITEM_NUMBER_WORDS:
            nxt = words[i + 1].text.lower().rstrip(".,") if i + 1 < n else ""
            if nxt in {"lugar", "place", "posicao", "posição"}:
                num = _ITEM_NUMBER_WORDS[token]
                boundary_idx = i

        if num is not None:
            candidates.append((num, words[boundary_idx].start))

    # Filtra mantendo apenas a sequencia descendente coerente.
    # Procuramos a maior subsequencia que comeca com um numero >= 3
    # e desce de 1 em 1 (10, 9, 8, ... ou 5, 4, 3, ...).
    boundaries: list[float] = []
    expected: int | None = None
    for num, ts in candidates:
        if expected is None:
            # So aceita iniciar com numero "alto" (>=3) para evitar falso match
            # em listas curtas com "1." espurio.
            if num >= 3:
                expected = num
                boundaries.append(ts)
                expected -= 1
        elif num == expected:
            boundaries.append(ts)
            expected -= 1
        # ignora qualquer numero fora de sequencia (ex.: "10 tons" depois de ja ter visto "9.")

    return boundaries


def suggest_cut_points(words: list[Word], language: str = "auto") -> list[float]:
    """Retorna lista de segundos sugeridos para encerrar cenas.

    Combina deteccao deterministica de fronteiras de item (regex) com
    sugestoes do Claude para sub-contextos dentro de cada item. As
    fronteiras de item sao SEMPRE incluidas, mesmo se o Claude falhar.
    """
    item_cuts = detect_item_boundaries(words)
    if item_cuts:
        _log(f"Detectadas {len(item_cuts)} fronteiras de item via regex: "
             f"{[f'{c:.1f}s' for c in item_cuts]}")

    llm_cuts = _suggest_cut_points_llm(words, language=language)

    merged = sorted(set(round(c, 2) for c in (item_cuts + llm_cuts)))
    # Remove duplicatas proximas (< 1.5s) preferindo a fronteira deterministica
    deduped: list[float] = []
    for c in merged:
        if deduped and c - deduped[-1] < 1.5:
            continue
        deduped.append(c)
    return deduped


def _suggest_cut_points_llm(words: list[Word], language: str = "auto") -> list[float]:
    """Pede ao Claude cortes adicionais. Retorna [] em caso de falha."""
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
        _log(f"Claude sugeriu {len(cuts)} pontos de corte adicionais.")
        return cuts
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _log(f"JSON invalido do Claude ({exc}) — ignorando sugestao do LLM.")
        return []
