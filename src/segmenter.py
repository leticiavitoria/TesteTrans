"""Divide a transcricao em cenas e prompts respeitando 8s/7s/148s."""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcribe import Word


BASE_DURATION = 8.0
EXT_DURATION = 7.0
MAX_EXT_PER_SCENE = 20
MAX_SCENE_DURATION = BASE_DURATION + MAX_EXT_PER_SCENE * EXT_DURATION  # 148s
SNAP_TOLERANCE = 1.0  # segundos de tolerancia no snap ao fim de palavra


@dataclass
class Prompt:
    start: float
    end: float
    kind: str  # "BASE" ou "EXT"
    text: str


@dataclass
class Scene:
    prompts: list[Prompt] = field(default_factory=list)


def _snap_word_end(words: list[Word], target: float, lo: float) -> tuple[float, int]:
    """Devolve (tempo_de_corte, indice_da_ultima_palavra_incluida).

    Procura o fim de palavra mais proximo de `target` dentro de [target-tol, target+tol]
    e estritamente apos `lo` (para garantir progresso). Se nao houver palavra
    nessa janela, corta exatamente em `target` (ou no fim da ultima palavra,
    o que for menor).
    """
    tol = SNAP_TOLERANCE
    best_idx = -1
    best_diff = float("inf")
    for i, w in enumerate(words):
        if w.end <= lo:
            continue
        if w.end > target + tol:
            break
        if w.end < target - tol:
            continue
        diff = abs(w.end - target)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    if best_idx >= 0:
        return words[best_idx].end, best_idx

    # Sem palavra na janela: corta no target, mas nao alem do fim da ultima palavra.
    last_end = words[-1].end if words else target
    cut = min(target, last_end)
    # encontra a ultima palavra cujo end <= cut
    last_included = -1
    for i, w in enumerate(words):
        if w.end <= cut:
            last_included = i
        else:
            break
    return cut, last_included


def _text_between(words: list[Word], start: float, end: float) -> str:
    chunk = [w.text for w in words if w.start >= start - 1e-6 and w.end <= end + 1e-6]
    return " ".join(chunk).strip()


def _pick_cut_in_window(
    cut_points: list[float], window_start: float, window_end: float
) -> float | None:
    """Retorna o cut_point dentro da janela, se houver."""
    for cp in cut_points:
        if window_start < cp <= window_end:
            return cp
    return None


def build_scenes(words: list[Word], cut_points: list[float] | None = None) -> list[Scene]:
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
        base_target = scene_start + BASE_DURATION
        base_end, _ = _snap_word_end(words, target=base_target, lo=scene_start)
        if base_end <= scene_start:
            base_end = min(scene_start + BASE_DURATION, audio_end)
        scene.prompts.append(
            Prompt(
                start=scene_start,
                end=base_end,
                kind="BASE",
                text=_text_between(words, scene_start, base_end),
            )
        )
        cursor = base_end

        # Se sugestao de corte cai dentro da BASE, fecha cena ja.
        forced_close = False
        if _pick_cut_in_window(cut_points, scene_start, base_end) is not None:
            forced_close = True

        # EXTs
        ext_count = 0
        while (
            not forced_close
            and ext_count < MAX_EXT_PER_SCENE
            and cursor < audio_end - 1e-6
            and (cursor - scene_start) < MAX_SCENE_DURATION - 1e-6
        ):
            ext_target = cursor + EXT_DURATION
            # respeita o limite duro de 148s
            scene_hard_limit = scene_start + MAX_SCENE_DURATION
            if ext_target > scene_hard_limit:
                ext_target = scene_hard_limit
            if ext_target > audio_end:
                ext_target = audio_end

            ext_end, _ = _snap_word_end(words, target=ext_target, lo=cursor)
            if ext_end <= cursor:
                ext_end = min(ext_target, audio_end)

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

            # Se o LLM sugeriu um corte dentro deste EXT, encerra a cena (com >=1 EXT).
            if _pick_cut_in_window(cut_points, scene.prompts[-1].start, ext_end) is not None:
                break

        scenes.append(scene)

    return scenes
