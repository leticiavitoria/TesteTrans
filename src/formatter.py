"""Formata as cenas no padrao SCENE/PROMPT solicitado."""

from __future__ import annotations

from .segmenter import Scene


def _mmss(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    s = int(round(t - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"


def format_scenes(scenes: list[Scene]) -> str:
    out: list[str] = []
    prompt_no = 0
    for i, scene in enumerate(scenes, start=1):
        if i > 1:
            out.append("")
        out.append(f"SCENE {i}")
        for p in scene.prompts:
            prompt_no += 1
            header = (
                f"PROMPT {prompt_no:03d} | {_mmss(p.start)} - {_mmss(p.end)} | {p.kind}"
            )
            out.append(header)
            if p.text:
                out.append(p.text)
    return "\n".join(out) + "\n"
