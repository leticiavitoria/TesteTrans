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
    "de personagem, de tempo, ou fim de uma ideia inteira) — NUNCA no meio de uma frase. "
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


def suggest_cut_points(words: list[Word], language: str = "auto") -> tuple[list[float], list[float]]:
    """Retorna (item_cuts, soft_cuts).

    - item_cuts: fronteiras OBRIGATORIAS entre itens da lista TOP-N (regex).
      Podem encurtar o prompt anterior abaixo do minimo, para que o BASE
      da nova cena comece exatamente no anuncio do item.
    - soft_cuts: sugestoes do Claude para sub-contextos. Inclui uma segunda
      passada focada em secoes que ainda estao acima de MAX_SECTION_DURATION
      apos a primeira chamada.
    """
    item_cuts = detect_item_boundaries(words)
    if item_cuts:
        _log(f"Detectadas {len(item_cuts)} fronteiras de item via regex: "
             f"{[f'{c:.1f}s' for c in item_cuts]}")

    llm_cuts = _suggest_cut_points_llm(words, language=language)

    # Remove cortes do LLM que estao muito proximos de uma fronteira de item
    # (a fronteira deterministica ja faz o mesmo papel).
    soft_cuts: list[float] = []
    for c in llm_cuts:
        if any(abs(c - ic) < 1.5 for ic in item_cuts):
            continue
        soft_cuts.append(c)

    # Segunda passada: identifica secoes que ainda excedem MAX_SECTION_DURATION
    # e pede ao Claude cortes coerentes especificamente nelas.
    extra = _resuggest_for_long_sections(
        words=words,
        item_cuts=item_cuts,
        soft_cuts=soft_cuts,
        language=language,
    )
    if extra:
        merged = soft_cuts + extra
        # Dedup contra item_cuts e contra cortes muito proximos entre si.
        soft_cuts = []
        for c in sorted(merged):
            if any(abs(c - ic) < 1.5 for ic in item_cuts):
                continue
            if soft_cuts and c - soft_cuts[-1] < 1.5:
                continue
            soft_cuts.append(c)

    return sorted(item_cuts), sorted(soft_cuts)


def _find_long_sections(
    audio_start: float, audio_end: float, boundaries: list[float]
) -> list[tuple[float, float]]:
    """Devolve pares (a, b) onde b - a > MAX_SECTION_DURATION."""
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
    """Para cada secao ainda > MAX_SECTION_DURATION, faz uma chamada focada ao
    Claude pedindo cortes coerentes dentro do trecho. Retorna a lista mesclada
    de novos cortes (em segundos absolutos).
    """
    if not words:
        return []
    audio_start = words[0].start
    audio_end = words[-1].end
    long_sections = _find_long_sections(
        audio_start, audio_end, item_cuts + soft_cuts
    )
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
                # So aceita cortes que estao DENTRO da secao com folga.
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
