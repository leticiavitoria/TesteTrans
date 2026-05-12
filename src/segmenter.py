"""Divide a transcricao em cenas e prompts.

Regras:
- BASE: sempre 8 segundos exatos.
- EXT: 7 segundos exatos cada, exceto a ULTIMA EXT da cena que pode ser menor.
- A ultima EXT de cada cena termina em PONTO FINAL (palavra terminada com ".").
- BASE de uma cena comeca no inicio de um novo item/ideia/contexto
  (item_cut ou soft_cut). Nao ha BASEs em pontos arbitrarios.
- Maximo de 20 EXTs por cena.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcribe import Word


BASE_DURATION = 8.0
EXT_DURATION = 7.0
MAX_EXT_PER_SCENE = 20
MAX_SCENE_DURATION = BASE_DURATION + MAX_EXT_PER_SCENE * EXT_DURATION  # 148s
PERIOD_LOOKBACK = 4.0  # quanto procuramos para tras de um boundary buscando um ponto final
EXT_ABSORB_THRESHOLD = 2.0  # se sobrar <= 2s apos uma EXT cheia, a ultima EXT absorve


@dataclass
class Prompt:
    start: float
    end: float
    kind: str  # "BASE" ou "EXT"
    text: str


@dataclass
class Scene:
    prompts: list[Prompt] = field(default_factory=list)


def _text_between(words: list[Word], start: float, end: float) -> str:
    chunk = [w.text for w in words if w.start >= start - 1e-6 and w.end <= end + 1e-6]
    return " ".join(chunk).strip()


def _find_period_end_before(words: list[Word], target: float, max_lookback: float) -> float | None:
    """Retorna o end time da ultima palavra terminada em '.' cujo end <= target,
    dentro da janela [target - max_lookback, target]. None se nao houver."""
    best: float | None = None
    lo = target - max_lookback
    for w in words:
        if w.end > target + 1e-6:
            break
        if w.end < lo:
            continue
        text = w.text.strip()
        if text.endswith(".") and not text.endswith(".."):
            best = w.end
    return best


def _find_period_end_in_range(words: list[Word], lo: float, hi: float) -> float | None:
    """Retorna o end time da palavra terminada em '.' mais proxima de `hi`
    dentro de [lo, hi]. None se nao houver."""
    best: float | None = None
    for w in words:
        if w.end < lo:
            continue
        if w.end > hi + 1e-6:
            break
        text = w.text.strip()
        if text.endswith(".") and not text.endswith(".."):
            best = w.end  # itera ate o fim, ficamos com o ultimo (mais proximo de hi)
    return best


def _find_period_end_after(words: list[Word], target: float, max_lookahead: float) -> float | None:
    """Retorna o end time da primeira palavra terminada em '.' cujo end >= target,
    dentro da janela [target, target + max_lookahead]."""
    hi = target + max_lookahead
    for w in words:
        if w.end < target - 1e-6:
            continue
        if w.end > hi:
            break
        text = w.text.strip()
        if text.endswith(".") and not text.endswith(".."):
            return w.end
    return None


def _adjust_boundary_to_period(
    words: list[Word], raw_boundary: float, prev_scene_start: float
) -> float:
    """Ajusta um boundary para o ponto final mais proximo ANTES dele.

    O boundary representa 'inicio de novo item/ideia' (ex.: a palavra '10.').
    O fim da cena anterior deve cair no ponto final imediatamente anterior.

    Garante que a cena anterior tenha pelo menos BASE_DURATION (8s).
    Se nao houver ponto final apropriado, devolve o proprio raw_boundary.
    """
    period = _find_period_end_before(words, raw_boundary, PERIOD_LOOKBACK)
    if period is not None and period >= prev_scene_start + BASE_DURATION:
        return period
    return raw_boundary


def build_scenes(
    words: list[Word],
    item_cuts: list[float] | None = None,
    soft_cuts: list[float] | None = None,
) -> list[Scene]:
    if not words:
        return []

    audio_start = words[0].start
    audio_end = words[-1].end

    raw_boundaries = sorted(set((item_cuts or []) + (soft_cuts or [])))

    # Constroi a lista de scene_starts ajustando cada boundary para o ponto final
    # imediatamente anterior. Filtra cenas curtas demais (< BASE_DURATION).
    scene_starts: list[float] = [audio_start]
    for b in raw_boundaries:
        if b <= scene_starts[-1] + BASE_DURATION - 1e-6:
            # Cena anterior nao caberia nem o BASE — ignora este boundary.
            continue
        adjusted = _adjust_boundary_to_period(words, b, scene_starts[-1])
        if adjusted <= scene_starts[-1] + BASE_DURATION - 1e-6:
            # Mesmo apos ajuste, ficou curta demais — usa o raw boundary.
            adjusted = b
        if adjusted <= scene_starts[-1] + 1e-6:
            continue
        scene_starts.append(adjusted)

    # Adiciona um sentinela com o fim do audio.
    scene_starts.append(audio_end)

    scenes: list[Scene] = []
    i = 0
    while i < len(scene_starts) - 1:
        scene_start = scene_starts[i]
        natural_end = scene_starts[i + 1]

        # Limita pela duracao maxima da cena (8 + 20*7 = 148s).
        max_end = scene_start + MAX_SCENE_DURATION
        if natural_end > max_end + 1e-6:
            # A secao entre os boundaries excede a duracao maxima. Procura um
            # ponto final perto do MEIO da secao para gerar duas cenas mais
            # balanceadas. Se a proxima cena ainda for longa demais, o loop
            # se chama recursivamente (proximo i) e divide de novo.
            section_len = natural_end - scene_start
            target = scene_start + section_len / 2.0
            min_split = scene_start + BASE_DURATION + EXT_DURATION  # garante BASE+1 EXT
            window = 25.0
            split = _find_period_end_in_range(
                words,
                lo=max(min_split, target - window),
                hi=min(max_end, target + window),
            )
            if split is None:
                # Alarga a janela de busca para todo o intervalo valido.
                split = _find_period_end_in_range(words, lo=min_split, hi=max_end)
            if split is None or split <= scene_start + BASE_DURATION:
                split = max_end  # ultimo recurso: corte duro
            # Garante que a proxima cena tambem tenha pelo menos BASE_DURATION.
            if natural_end - split < BASE_DURATION:
                split = max(min_split, natural_end - BASE_DURATION)
            scene_end = split
            scene_starts.insert(i + 1, split)
        else:
            scene_end = natural_end

        if scene_end > audio_end:
            scene_end = audio_end

        # Para a ULTIMA cena (que termina em audio_end), ajusta para o ponto final
        # mais proximo de audio_end, se houver.
        is_last = (i == len(scene_starts) - 2)
        if is_last:
            period = _find_period_end_before(words, audio_end, PERIOD_LOOKBACK)
            if period is not None and period > scene_start + BASE_DURATION:
                scene_end = period

        scene = Scene()

        # BASE: 8s exatos. Excecao: se a cena toda cabe em ate ~10s
        # (BASE_DURATION + EXT_ABSORB_THRESHOLD), o BASE absorve tudo para
        # nao gerar uma micro-EXT. Tambem limitado por scene_end em cenas curtas.
        scene_total = scene_end - scene_start
        if scene_total <= BASE_DURATION + EXT_ABSORB_THRESHOLD + 1e-6:
            base_end = scene_end
        else:
            base_end = scene_start + BASE_DURATION
        scene.prompts.append(
            Prompt(
                start=scene_start,
                end=base_end,
                kind="BASE",
                text=_text_between(words, scene_start, base_end),
            )
        )
        cursor = base_end

        # EXTs de 7s, exceto a ultima que vai ate scene_end.
        # Se sobrariam <= EXT_ABSORB_THRESHOLD apos uma EXT cheia, a ULTIMA EXT
        # absorve o residuo (vira EXT de 7 ate 7+THRESHOLD segundos).
        ext_count = 0
        while cursor < scene_end - 1e-6 and ext_count < MAX_EXT_PER_SCENE:
            remaining = scene_end - cursor
            if remaining <= EXT_DURATION + EXT_ABSORB_THRESHOLD + 1e-6:
                # Ultima EXT — vai ate scene_end (7s..9s ou menor).
                ext_end = scene_end
            else:
                # EXT cheia de 7s.
                ext_end = cursor + EXT_DURATION
            scene.prompts.append(
                Prompt(
                    start=cursor,
                    end=ext_end,
                    kind="EXT",
                    text=_text_between(words, cursor, ext_end),
                )
            )
            cursor = ext_end
            ext_count += 1

        scenes.append(scene)
        i += 1

    # Descarta cena final vazia ou degenerada (BASE com <1s e sem texto).
    if scenes:
        last = scenes[-1]
        if last.prompts:
            p0 = last.prompts[0]
            if p0.end - p0.start < 1.0 and not p0.text.strip():
                scenes.pop()

    return scenes
