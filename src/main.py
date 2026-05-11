"""CLI: transcreve audio e imprime/salva o relatorio em cenas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .formatter import format_scenes
from .scene_planner import suggest_cut_points
from .segmenter import build_scenes
from .transcribe import transcribe


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Transcreve audio com Whisper Large V3 e divide em cenas."
    )
    parser.add_argument("audio", help="Caminho para o arquivo de audio.")
    parser.add_argument(
        "-o",
        "--output",
        help="Arquivo de saida .txt. Se omitido, imprime no stdout.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Pula a chamada ao Claude e usa apenas a heuristica.",
    )
    args = parser.parse_args(argv)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Arquivo nao encontrado: {audio_path}", file=sys.stderr)
        return 1

    print(f"Transcrevendo {audio_path}...", file=sys.stderr)
    words, language = transcribe(str(audio_path))
    if not words:
        print("Nenhuma palavra detectada na transcricao.", file=sys.stderr)
        return 2
    print(f"Idioma detectado: {language}. Palavras: {len(words)}.", file=sys.stderr)

    if args.no_llm:
        cut_points: list[float] = []
    else:
        print("Consultando Claude para sugerir cortes de cena...", file=sys.stderr)
        cut_points = suggest_cut_points(words, language=language)
        print(f"Cortes sugeridos: {len(cut_points)}.", file=sys.stderr)

    scenes = build_scenes(words, cut_points=cut_points)
    output = format_scenes(scenes)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Relatorio salvo em {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
