# TesteTrans

Transcreve um áudio com Whisper Large V3 e divide a transcrição em **cenas**
respeitando uma estrutura rígida de prompts:

- 1 prompt **BASE** com aproximadamente **8 segundos**
- até 20 prompts **EXT** com aproximadamente **7 segundos** cada
- duração máxima por cena: `8 + 20 * 7 = 148 segundos`

Os limites de cada prompt são "encaixados" (snap) ao **fim da palavra mais
próxima**, garantindo que nenhuma palavra seja cortada ao meio. Os pontos de
quebra de cena são sugeridos por **Claude Sonnet 4.6** (mudança de tópico,
encerramento de ideia) e validados pela heurística determinística.

## Instalação

```bash
pip install -r requirements.txt
cp .env.example .env
# edite .env e coloque sua ANTHROPIC_API_KEY
```

A primeira execução baixa o modelo `large-v3` do Whisper (~3 GB).

## Uso

```bash
python -m src.main caminho/para/audio.mp3 -o saida.txt
```

Saída no formato:

```
SCENE 1
PROMPT 001 | 00:00 - 00:08 | BASE
texto da janela base
PROMPT 002 | 00:08 - 00:15 | EXT
texto da continuação
...

SCENE 2
PROMPT 005 | 00:29 - 00:37 | BASE
...
```

## Notas

- Sem GPU, transcrever áudios longos pode demorar. Use `--compute-type int8`
  (passe via env `WHISPER_COMPUTE_TYPE=int8`) para acelerar em CPU.
- Se a chamada ao Claude falhar, o programa continua usando apenas a
  heurística determinística (sem sugestões semânticas).
