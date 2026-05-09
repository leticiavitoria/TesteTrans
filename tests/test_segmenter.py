"""Testes para a logica deterministica de segmentacao."""

from __future__ import annotations

from src.segmenter import (
    BASE_DURATION,
    EXT_DURATION,
    MAX_EXT_PER_SCENE,
    MAX_SCENE_DURATION,
    build_scenes,
)
from src.transcribe import Word


def make_words(total_seconds: float, word_dur: float = 0.5) -> list[Word]:
    words = []
    t = 0.0
    i = 0
    while t + word_dur <= total_seconds + 1e-9:
        words.append(Word(start=t, end=t + word_dur, text=f"w{i}"))
        t += word_dur
        i += 1
    return words


def test_short_audio_one_scene_one_base():
    words = make_words(5.0)
    scenes = build_scenes(words)
    assert len(scenes) == 1
    assert len(scenes[0].prompts) == 1
    assert scenes[0].prompts[0].kind == "BASE"
    assert scenes[0].prompts[0].end <= 5.0 + 1e-6


def test_no_word_split_snap_to_word_end():
    words = make_words(20.0, word_dur=0.5)
    scenes = build_scenes(words)
    for scene in scenes:
        for p in scene.prompts:
            # cada borda deve coincidir com um final de palavra ou com o fim do audio
            assert any(abs(p.end - w.end) < 1e-6 for w in words) or abs(
                p.end - words[-1].end
            ) < 1e-6


def test_scene_never_exceeds_148s():
    words = make_words(600.0, word_dur=0.4)
    scenes = build_scenes(words)
    for scene in scenes:
        dur = scene.prompts[-1].end - scene.prompts[0].start
        assert dur <= MAX_SCENE_DURATION + 1e-6
        assert scene.prompts[0].kind == "BASE"
        assert sum(1 for p in scene.prompts if p.kind == "EXT") <= MAX_EXT_PER_SCENE


def test_llm_cut_point_closes_scene_early():
    words = make_words(200.0, word_dur=0.5)
    # forca corte aos 30s
    scenes = build_scenes(words, cut_points=[30.0])
    first = scenes[0]
    last_end = first.prompts[-1].end
    # primeira cena deve terminar perto dos 30s, bem antes de 148s
    assert last_end < 60.0
    assert first.prompts[0].kind == "BASE"


def test_full_coverage_no_gaps():
    words = make_words(120.0, word_dur=0.5)
    scenes = build_scenes(words)
    # comeca no inicio
    assert abs(scenes[0].prompts[0].start - words[0].start) < 1e-6
    # termina no fim do audio (ultima palavra)
    assert abs(scenes[-1].prompts[-1].end - words[-1].end) < 1.0
    # sem buracos entre prompts consecutivos da mesma cena
    for scene in scenes:
        for a, b in zip(scene.prompts, scene.prompts[1:]):
            assert abs(a.end - b.start) < 1e-6


def test_durations_close_to_targets():
    words = make_words(60.0, word_dur=0.5)
    scenes = build_scenes(words)
    base = scenes[0].prompts[0]
    assert abs((base.end - base.start) - BASE_DURATION) <= 1.0
    if len(scenes[0].prompts) > 1:
        ext = scenes[0].prompts[1]
        assert abs((ext.end - ext.start) - EXT_DURATION) <= 1.0
