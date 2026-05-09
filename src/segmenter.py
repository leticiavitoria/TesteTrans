"""Divide a transcricao em cenas e prompts. Cada prompt dura entre 7 e 8 segundos.

Excecao: prompts imediatamente antes do anuncio de um novo item da lista
podem ser encurtados (ate ~3s) para que o BASE da nova cena comece exatamente
no anuncio (ex.: "10. Ulfberht...", "Number 8 Lycurgus...").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcribe import Word


PROMPT_MIN = 7.0
PROMPT_MAX = 8.0
ITEM_BOUNDARY_MIN = 3.0  # prompt pode encurtar ate isso para alinhar com item_cut
ITEM_BOUNDARY_MAX = 10.0  # ou pode estender ate isso, se item_cut cair entre janelas
MAX_EXT_PER_SCENE = 20
MAX_SCENE_DURATION = PROMPT_MAX + MAX_EXT_PER_SCENE * PROMPT_MAX  # 168s
MIN_SCENE_DURATION = PROMPT_MIN  # soft cuts mais perto do inicio que isso sao ignorados


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
    """Fim de palavra mais proximo do meio de [win_start, win_end], apos `lo`.

    Se nenhuma palavra terminar na janela, devolve win_end.
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

    last_end = words[-1].end if words else win_end
    return min(win_end, last_end)


def _snap_word_end_near(words: list[Word], lo: float, target: float, tol: float = 0.6) -> float:
    """Fim de palavra mais proximo de `target` dentro de [target-tol, target+tol], apos `lo`.

    Se nao houver, devolve `target` mesmo (truncado em audio_end pelo caller).
    """
    best_end = -1.0
    best_diff = float("inf")
    for w in words:
        if w.end <= lo:
            continue
        if w.end > target + tol:
            break
        if w.end < target - tol:
            continue
        diff = abs(w.end - target)
        if diff < best_diff:
            best_diff = diff
            best_end = w.end
    return best_end if best_end > 0 else target


def _text_between(words: list[Word], start: float, end: float) -> str:
    chunk = [w.text for w in words if w.start >= start - 1e-6 and w.end <= end + 1e-6]
    return " ".join(chunk).strip()


def _next_after(cuts: list[float], lo: float) -> float | None:
    for c in cuts:
        if c > lo + 1e-6:
            return c
    return None


def _build_prompt_end(
    words: list[Word],
    prompt_start: float,
    audio_end: float,
    item_cuts: list[float],
    soft_cuts: list[float],
    scene_start: float,
) -> tuple[float, bool]:
    """Decide onde este prompt termina e se a cena deve fechar depois dele.

    Regras:
    - Se um item_cut estiver em [prompt_start + ITEM_BOUNDARY_MIN, prompt_start + PROMPT_MAX],
      o prompt e truncado proximo ao item_cut (snap ao fim de palavra mais proximo)
      e a cena fecha. Permite prompt curto (3-8s) para alinhar BASE seguinte ao item.
    - Caso contrario, o prompt termina dentro de [start+7, start+8] (snap normal).
    - Se um soft_cut cair em [start+7, start+8], a cena fecha apos este prompt.
    """
    win_start = prompt_start + PROMPT_MIN
    win_end = prompt_start + PROMPT_MAX

    # 1) Item boundary em [prompt_start+3, prompt_start+10] => alinha este prompt
    #    com o item_cut (encurtando ate 3s ou estendendo ate 10s) e fecha cena.
    #    Ignora item_cuts a menos de 1s do inicio (ja foram consumidos pela cena anterior).
    item_cut = _next_after(item_cuts, prompt_start + 1.0)
    if (
        item_cut is not None
        and prompt_start + ITEM_BOUNDARY_MIN <= item_cut <= prompt_start + ITEM_BOUNDARY_MAX
    ):
        end = _snap_word_end_near(words, lo=prompt_start, target=item_cut, tol=0.6)
        if end > audio_end:
            end = audio_end
        if end <= prompt_start:
            end = min(item_cut, audio_end)
        return end, True

    # 2) Snap normal dentro da janela 7-8s.
    end = _snap_word_end_in_range(words, lo=prompt_start, win_start=win_start, win_end=win_end)
    if end > audio_end:
        end = audio_end
    if end <= prompt_start:
        end = min(prompt_start + PROMPT_MAX, audio_end)

    # 3) Soft cut dentro da janela => fecha cena apos este prompt.
    close_after = False
    soft = _next_after(soft_cuts, scene_start + MIN_SCENE_DURATION)
    if soft is not None and prompt_start < soft <= end + 0.5:
        close_after = True

    return end, close_after


def build_scenes(
    words: list[Word],
    item_cuts: list[float] | None = None,
    soft_cuts: list[float] | None = None,
) -> list[Scene]:
    """Constroi cenas. Veja docstring do modulo para regras."""
    if not words:
        return []
    item_cuts = sorted(item_cuts or [])
    soft_cuts = sorted(soft_cuts or [])

    scenes: list[Scene] = []
    cursor = words[0].start
    audio_end = words[-1].end

    while cursor < audio_end - 1e-6:
        scene = Scene()
        scene_start = cursor

        base_end, close_after_base = _build_prompt_end(
            words,
            prompt_start=scene_start,
            audio_end=audio_end,
            item_cuts=item_cuts,
            soft_cuts=soft_cuts,
            scene_start=scene_start,
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
                item_cuts=item_cuts,
                soft_cuts=soft_cuts,
                scene_start=scene_start,
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
