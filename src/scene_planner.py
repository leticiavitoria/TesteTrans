"""Pede ao Claude que sugira pontos de quebra de cena (em segundos)."""

from __future__ import annotations

import json
import os
import re
import sys

from .transcribe import Word


CLAUDE_MODEL = "claude-opus-4-6"
MAX_SECTION_DURATION = 148.0  # = BASE_DURATION + MAX_EXT_PER_SCENE * EXT_DURATION
TARGET_MAX_SECTION = 140.0    # alvo recomendado ao Claude (folga de 8s)
MIN_SCENE_DURATION = 8.0      # = BASE_DURATION; precisa bater com segmenter
PERIOD_LOOKBACK = 8.0         # janela para tras ao snap de cortes em ponto final
PERIOD_LOOKAHEAD = 5.0        # janela para frente ao snap de cortes em ponto final
SENTENCE_ENDERS = (".", "!", "?")


SYSTEM_PROMPT = (
    "Voce recebe a transcricao de um audio com timestamps por palavra. "
    "O audio costuma ser um video de lista TOP-N (Top 10, Top 5, etc). "
    "Sua tarefa e identificar os instantes onde cada CENA narrativa termina.\n\n"
    "REGRA CRITICA E INVIOLAVEL: nenhuma secao entre dois cut_points consecutivos "
    f"(incluindo as fronteiras de item) pode durar mais de {TARGET_MAX_SECTION:.0f} segundos. "
    "Se um item da lista TOP-N for mais longo que isso, voce DEVE adicionar cut_points "
    "intermediarios em pontos de troca clara de sub-contexto narrativo (mudanca de foco, "
    "de quem fala, de tempo/local da narrativa, fim de uma ideia inteira).\n\n"
    "CONCEITO: cada cena comeca com um prompt BASE e e seguida de extensoes (EXT) "
    "que detalham o MESMO mini-contexto. Quando o mini-contexto muda, uma nova cena "
    "(novo BASE) deve comecar — mesmo que ainda estejamos dentro do mesmo item da lista. "
    "Pense em cada cena como uma 'unidade visual' que ilustraria um momento especifico da narrativa.\n\n"
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
    "(d) misterio/teorias concorrentes, "
    "(e) implicacao/conclusao do item. "
    "Itens longos devem virar 3 a 6 cenas. Itens curtos podem ter 1 a 2 cenas.\n"
    "4. O FECHAMENTO/OUTRO (recapitulacao, CTA, despedida apos o item 1) "
    "deve ser pelo menos uma cena propria, possivelmente quebrada em recapitulacao + CTA.\n\n"
    "REGRAS GERAIS:\n"
    "- Cada cut_point DEVE coincidir com o FIM de uma frase (palavra terminada em '.', '!' ou '?'). "
    "Use os timestamps das palavras na transcricao — o cut_point e o instante final (end) da palavra "
    "que encerra a frase. Cortes no meio de uma frase serao DESCARTADOS automaticamente.\n"
    "- Nao corte no meio de uma frase. Prefira o fim da sentenca antes do novo mini-contexto.\n"
    "- Cenas podem ter duracoes muito diferentes. NAO force tamanhos similares.\n"
    "- O numero TOTAL de cenas deve ser MAIOR que (intro + N itens + outro). "
    "Para um Top 10, espere algo como 25 a 50 cenas no total, nao apenas 12.\n"
    "- Quando estiver em duvida entre cortar ou nao cortar dentro de um item, PREFIRA cortar — "
    "cenas mais granulares sao melhores que cenas longas demais.\n\n"
    "Responda APENAS com JSON valido no formato: "
    '{\"cut_points_seconds\": [12.4, 31.7, ...]} '
    "onde cada numero e o instante (em segundos) onde uma cena DEVE terminar."
)


LONG_SECTION_SYSTEM_PROMPT = (
    "Voce recebe um TRECHO da transcricao de um audio que ainda esta longo demais "
    f"(mais de {MAX_SECTION_DURATION:.0f} segundos). Sua tarefa e propor 1 ou mais "
    "cut_points DENTRO desse trecho que dividam a narrativa em sub-cenas coerentes, "
    f"de forma que NENHUMA sub-secao resultante exceda {TARGET_MAX_SECTION:.0f} segundos.\n\n"
    "Cada cut_point deve cair em uma troca real de sub-contexto narrativo (mudanca de foco, "
    "de personagem, de tempo, ou fim de uma ideia inteira) e DEVE coincidir com o fim de uma "
    "frase (palavra terminada em '.', '!' ou '?'). NUNCA corte no meio de uma frase. "
    "Use o conteudo do texto como guia: pense em cada sub-cena como uma 'unidade visual' "
    "distinta. NAO retorne os limites externos do trecho — somente os cortes INTERNOS.\n\n"
    "Responda APENAS com JSON valido no formato: "
    '{\"cut_points_seconds\": [12.4, 31.7, ...]} '
    "onde cada numero e um instante (em segundos absolutos no audio original) onde "
    "uma sub-cena DEVE terminar."
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

    Estrategia: procurar uma SEQUENCIA DESCENDENTE de numeros (10 -> 9 -> 8 ... 1)
    em contexto de anuncio de item. Aceita "Number nine" por extenso e tolera
    lacunas (apenas exige descendencia estrita).
    """
    n = len(words)
    candidates: list[tuple[int, float]] = []

    for i, w in enumerate(words):
        token = w.text.strip().rstrip(".,;:!?\"'").lower()
        num: int | None = None
        boundary_idx = i

        # Padrao "Number 10", "Number nine", "Numero dez"
        if token in {"number", "numero", "número"} and i + 1 < n:
            nxt_token = words[i + 1].text.strip().rstrip(".,;:!?\"'").lower()
            cand: int | None = None
            m = re.match(r"^(\d{1,2})$", nxt_token)
            if m:
                cand = int(m.group(1))
            elif nxt_token in _ITEM_NUMBER_WORDS:
                cand = _ITEM_NUMBER_WORDS[nxt_token]
            if cand is not None and 1 <= cand <= 20:
                num = cand
                boundary_idx = i

        # Padrao "10.", "9.", etc.
        if num is None:
            m = re.match(r"^(\d{1,2})\.?$", token)
            if m:
                cand = int(m.group(1))
                if 1 <= cand <= 20:
                    nxt_raw = words[i + 1].text.strip() if i + 1 < n else ""
                    nxt_first = nxt_raw[:1]
                    if nxt_first.isalpha() and nxt_first.isupper():
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

    boundaries: list[float] = []
    last_accepted: int | None = None
    for num, ts in candidates:
        if last_accepted is None:
            if num >= 3:
                last_accepted = num
                boundaries.append(ts)
        elif num < last_accepted:
            last_accepted = num
            boundaries.append(ts)

    return boundaries


def _is_sentence_end(word_text: str) -> bool:
    t = word_text.strip()
    if not t or t.endswith(".."):
        return False
    return t.endswith(SENTENCE_ENDERS)


def _nearest_period_end(
    words: list[Word], target: float, lookback: float, lookahead: float
) -> float | None:
    """Retorna o end-time da palavra terminada em '.', '!' ou '?' mais proxima
    de `target` dentro de [target - lookback, target + lookahead]."""
    lo = target - lookback
    hi = target + lookahead
    best: float | None = None
    best_dist = float("inf")
    for w in words:
        if w.end < lo - 1e-6:
            continue
        if w.end > hi + 1e-6:
            break
        if not _is_sentence_end(w.text):
            continue
        d = abs(w.end - target)
        if d < best_dist:
            best_dist = d
            best = w.end
    return best


def suggest_cut_points(words: list[Word], language: str = "auto") -> list[float]:
    """Retorna uma lista unica de cut_points (em segundos), ja snapados para
    o fim da frase mais proxima. Combina:

    - fronteiras OBRIGATORIAS de item da lista TOP-N (regex deterministico),
      snapadas para o ponto final imediatamente ANTES do anuncio;
    - sugestoes do Claude para sub-contextos (snapadas para o ponto final
      mais proximo; descartadas quando nao ha ponto final na janela);
    - cortes da segunda passada do Claude para secoes ainda > MAX_SECTION_DURATION.

    Cortes suaves a menos de MIN_SCENE_DURATION de qualquer fronteira de item
    sao descartados — o anuncio do item sempre vence.
    """
    if not words:
        return []

    item_cuts_raw = detect_item_boundaries(words)
    if item_cuts_raw:
        _log(f"Detectadas {len(item_cuts_raw)} fronteiras de item via regex: "
             f"{[f'{c:.1f}s' for c in item_cuts_raw]}")

    # Snap item cuts para o ponto final imediatamente ANTES do anuncio.
    item_cuts: list[float] = []
    for ic in item_cuts_raw:
        snapped = _nearest_period_end(words, ic, lookback=PERIOD_LOOKBACK, lookahead=0.5)
        item_cuts.append(snapped if snapped is not None else ic)
    item_cuts = sorted(item_cuts)

    llm_cuts = _suggest_cut_points_llm(words, language=language)

    # Snap soft cuts; descarta os sem ponto final na janela.
    soft_cuts: list[float] = []
    dropped = 0
    for c in llm_cuts:
        snapped = _nearest_period_end(words, c, lookback=PERIOD_LOOKBACK, lookahead=PERIOD_LOOKAHEAD)
        if snapped is None:
            dropped += 1
            continue
        soft_cuts.append(snapped)
    if dropped:
        _log(f"{dropped} corte(s) suave(s) descartado(s): sem ponto final num raio de "
             f"{PERIOD_LOOKBACK:.0f}s/{PERIOD_LOOKAHEAD:.0f}s.")

    # Item cuts dominam: descarta soft cuts muito proximos.
    soft_cuts = _drop_close_to_items(soft_cuts, item_cuts)

    # Segunda passada para secoes ainda > MAX_SECTION_DURATION.
    extra = _resuggest_for_long_sections(
        words=words,
        item_cuts=item_cuts,
        soft_cuts=soft_cuts,
        language=language,
    )
    if extra:
        for e in extra:
            snapped = _nearest_period_end(words, e, lookback=PERIOD_LOOKBACK, lookahead=PERIOD_LOOKAHEAD)
            if snapped is None:
                continue
            soft_cuts.append(snapped)
        soft_cuts = _drop_close_to_items(soft_cuts, item_cuts)

    # Dedup final.
    soft_cuts = sorted(set(soft_cuts))
    deduped: list[float] = []
    for c in soft_cuts:
        if deduped and c - deduped[-1] < MIN_SCENE_DURATION:
            continue
        deduped.append(c)
    soft_cuts = deduped

    return sorted(set(item_cuts) | set(soft_cuts))


def _drop_close_to_items(soft_cuts: list[float], item_cuts: list[float]) -> list[float]:
    out: list[float] = []
    for s in soft_cuts:
        if any(abs(s - i) < MIN_SCENE_DURATION for i in item_cuts):
            continue
        out.append(s)
    return out


def _find_long_sections(
    audio_start: float, audio_end: float, boundaries: list[float]
) -> list[tuple[float, float]]:
    starts = sorted({audio_start, audio_end, *boundaries})
    return [
        (a, b)
        for a, b in zip(starts, starts[1:])
        if b - a > MAX_SECTION_DURATION + 1e-6
    ]


def _resuggest_for_long_sections(
    words: list[Word],
    item_cuts: list[float],
    soft_cuts: list[float],
    language: str,
) -> list[float]:
    """Segunda passada focada em trechos ainda > MAX_SECTION_DURATION."""
    if not words:
        return []
    audio_start = words[0].start
    audio_end = words[-1].end
    long_sections = _find_long_sections(audio_start, audio_end, item_cuts + soft_cuts)
    if not long_sections:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    try:
        from anthropic import Anthropic
    except ImportError:
        return []

    _log(f"{len(long_sections)} secao(oes) ainda > {MAX_SECTION_DURATION:.0f}s — "
         f"pedindo segunda passada ao Claude.")

    client = Anthropic(api_key=api_key)
    new_cuts: list[float] = []

    for sec_start, sec_end in long_sections:
        section_words = [w for w in words if w.start >= sec_start - 1e-6 and w.end <= sec_end + 1e-6]
        if not section_words:
            continue

        transcript_lines = []
        for w in section_words:
            mm = int(w.start // 60)
            ss = w.start - mm * 60
            transcript_lines.append(f"[{mm:02d}:{ss:05.2f}] {w.text}")
        transcript = "\n".join(transcript_lines)

        section_len = sec_end - sec_start
        user_msg = (
            f"Idioma: {language}. Trecho de {section_len:.1f}s ({sec_start:.1f}s a {sec_end:.1f}s).\n"
            f"Transcricao do trecho:\n{transcript}\n\n"
            f"Devolva o JSON com cortes INTERNOS para que nenhuma sub-secao "
            f"exceda {TARGET_MAX_SECTION:.0f}s."
        )

        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": LONG_SECTION_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:
            _log(f"Erro na segunda passada para [{sec_start:.1f}, {sec_end:.1f}]: {exc}")
            continue

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
            for c in data.get("cut_points_seconds", []):
                if not isinstance(c, (int, float)):
                    continue
                cf = float(c)
                if sec_start + 5.0 <= cf <= sec_end - 5.0:
                    new_cuts.append(cf)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if new_cuts:
        _log(f"Segunda passada devolveu {len(new_cuts)} corte(s) adicional(is).")
    return new_cuts


def _suggest_cut_points_llm(words: list[Word], language: str = "auto") -> list[float]:
    """Pede ao Claude cortes adicionais. Retorna [] em caso de falha."""
    if not words:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log("ANTHROPIC_API_KEY nao definida — pulando Claude.")
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
            model=CLAUDE_MODEL,
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
