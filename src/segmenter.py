"""Divide a transcricao em cenas e prompts. Cada prompt dura entre 7 e 8 segundos."""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcribe import Word


PROMPT_MIN = 7.0
PROMPT_MAX = 8.0
MAX_EXT_PER_SCENE = 20
MAX_SCENE_DURATION = PROMPT_MAX + MAX_EXT_PER_SCENE * PROMPT_MAX  # 168s
MIN_SCENE_DURATION = PROMPT_MIN  # cut_points mais perto que isso do inicio da cena sao ignorados


@dataclass
class Prompt:
    start: float
    end: float
    kind: str  # "BASE" ou "EXT"
    text: str


@dataclass
class Scene:
    prompts: list[Prompt] = field(default_factory=list)


def _snap_word_end_in_range(
    words: list[Word], lo: float, win_start: float, win_end: float
) -> float:
    """Devolve o melhor instante para terminar um prompt cujo inicio e `lo`.

    Procura o fim de palavra dentro de [win_start, win_end] que esteja mais
    proximo do meio da janela (ou seja, ~7.5s apos o inicio). Se nenhuma
    palavra terminar nessa janela, devolve win_end (corte forcado no maximo).
    """
    target = (win_start + win_end) / 2.0
    best_end = -1.0
    best_diff = float("inf")
    for w in words:
        if w.end <= lo:
            continue
        if w.end > win_end + 1e-6:
            break
        if w.end < win_start - 1e-6:
            continue
        diff = abs(w.end - target)
        if diff < best_diff:
            best_diff = diff
            best_end = w.end

    if best_end > 0:
        return best_end

    # Sem fim de palavra na janela: corta no win_end (limite duro de 8s).
    last_end = words[-1].end if words else win_end
    return min(win_end, last_end)


def _text_between(words: list[Word], start: float, end: float) -> str:
    chunk = [w.text for w in words if w.start >= start - 1e-6 and w.end <= end + 1e-6]
    return " ".join(chunk).strip()


def _next_cut_after(cut_points: list[float], lo: float, min_offset: float = 0.0) -> float | None:
    """Retorna o primeiro cut_point estritamente apos `lo + min_offset`, ou None."""
    threshold = lo + max(min_offset, 1e-6)
    for cp in cut_points:
        if cp > threshold:
            return cp
    return None


def _build_prompt_end(
    words: list[Word],
    prompt_start: float,
    audio_end: float,
    cut_points: list[float],
    scene_start: float,
    is_base: bool,
) -> tuple[float, bool]:
    """Decide onde este prompt (BASE ou EXT) termina.

    Retorna (end_time, close_scene_after). O prompt sempre dura entre PROMPT_MIN
    e PROMPT_MAX segundos, exceto se o audio acabar antes.
    """
    win_start = prompt_start + PROMPT_MIN
    win_end = prompt_start + PROMPT_MAX

    close_after = False
    cp = _next_cut_after(cut_points, scene_start, min_offset=MIN_SCENE_DURATION)
    if cp is not None and prompt_start < cp <= win_end:
        # Cut_point cai dentro da janela do prompt — fecha cena no fim deste prompt
        # (mas o prompt ainda dura entre 7 e 8s, snap dentro da janela).
        close_after = True
    elif cp is not None and cp <= prompt_start:
        # Cut_point ja foi atingido em prompts anteriores; tambem fecha.
        close_after = True

    end = _snap_word_end_in_range(words, lo=prompt_start, win_start=win_start, win_end=win_end)
    if end > audio_end:
        end = audio_end
    if end <= prompt_start:
        end = min(prompt_start + PROMPT_MAX, audio_end)

    return end, close_after


def build_scenes(words: list[Word], cut_points: list[float] | None = None) -> list[Scene]:
    """Constroi cenas. Cada prompt dura 7-8s. Cut_points fecham a cena no
    proximo prompt cuja janela [start+7, start+8] contem o cut_point.
    """
    if not words:
        return []
    cut_points = sorted(cut_points or [])

    scenes: list[Scene] = []
    cursor = words[0].start
    audio_end = words[-1].end

    while cursor < audio_end - 1e-6:
        scene = Scene()
        scene_start = cursor

        # BASE
        base_end, close_after_base = _build_prompt_end(
            words,
            prompt_start=scene_start,
            audio_end=audio_end,
            cut_points=cut_points,
            scene_start=scene_start,
            is_base=True,
        )
        scene.prompts.append(
            Prompt(
                start=scene_start,
                end=base_end,
                kind="BASE",
                text=_text_between(words, scene_start, base_end),
            )
        )
        cursor = base_end

        # EXTs
        ext_count = 0
        while (
            not close_after_base
            and ext_count < MAX_EXT_PER_SCENE
            and cursor < audio_end - 1e-6
            and (cursor - scene_start) < MAX_SCENE_DURATION - 1e-6
        ):
            ext_end, close_after = _build_prompt_end(
                words,
                prompt_start=cursor,
                audio_end=audio_end,
                cut_points=cut_points,
                scene_start=scene_start,
                is_base=False,
            )
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

            if close_after:
                break

        scenes.append(scene)

    return scenes
