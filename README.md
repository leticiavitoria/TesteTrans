# TesteTrans

Transcreve um áudio com **Whisper Large V3** (via `faster-whisper`) e divide a
transcrição em **cenas** seguindo regras rígidas de duração e fronteiras
narrativas. Pensado para roteiros de vídeos TOP-N (Top 10, Top 5, etc).

## Como o sistema decide as cenas

Cada cena tem:

- 1 prompt **BASE** de **8 segundos exatos**
- até 20 prompts **EXT** de **7 segundos exatos cada**
- a **última EXT** pode ser menor ou um pouco maior (absorve resíduo de até 2s)
- a **última EXT termina sempre em ponto final** (não corta no meio de frase)
- duração máxima da cena: `8 + 20*7 = 148 segundos`

Cada cena **começa numa fronteira semântica**, vinda de duas fontes:

1. **Detecção determinística via regex** (`detect_item_boundaries`): identifica
   anúncios de itens da lista (`10.`, `9.`, `Number 8`, "décimo lugar", etc.)
   exigindo sequência descendente coerente para evitar falsos positivos
   ("10 tons", "1.000 years", "2.8 billion").
2. **Sugestões do Claude Opus 4.6** (`scene_planner.py`): identifica trocas de
   sub-contexto dentro de cada item (descrição → descoberta → análise →
   teorias → conclusão).

Se alguma seção entre fronteiras ainda passar de 148s, o sistema faz uma
**segunda chamada focada** ao Claude pedindo cortes coerentes dentro daquele
trecho específico. Como último recurso, há um fallback determinístico que
busca um ponto final próximo do meio da seção.

## Estrutura do projeto

```
TesteTrans/
├── README.md
├── requirements.txt
├── .env.example          # template para a ANTHROPIC_API_KEY
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py           # CLI (python -m src.main)
│   ├── transcribe.py     # wrapper sobre faster-whisper
│   ├── scene_planner.py  # detecção regex + chamadas ao Claude
│   ├── segmenter.py      # constrói cenas e prompts
│   └── formatter.py      # formata a saída SCENE/PROMPT
└── tests/
    └── test_segmenter.py
```

## Instalação local

```bash
pip install -r requirements.txt
cp .env.example .env
# edite .env e coloque sua ANTHROPIC_API_KEY
```

A primeira execução baixa o modelo `large-v3` do Whisper (~3 GB).

## Uso local

```bash
python -m src.main caminho/para/audio.mp3 -o saida.txt
```

Variáveis de ambiente úteis:

- `ANTHROPIC_API_KEY` — chave da Anthropic (sem ela, só o regex detecta itens).
- `WHISPER_DEVICE` — `cuda`, `cpu` ou `auto` (padrão).
- `WHISPER_COMPUTE_TYPE` — `float16` (GPU), `int8` (CPU rápido), `auto`.

Flags da CLI:

- `-o, --output` — arquivo de saída (se omitido, imprime no stdout).
- `--no-llm` — pula o Claude e usa só a heurística determinística.

## Uso no Google Colab

1. Em `Runtime → Change runtime type → GPU (T4)`.
2. Em **Secrets** (ícone de chave à esquerda), adicione `ANTHROPIC_API_KEY`.
3. Numa célula:

```python
!pip install -q faster-whisper anthropic python-dotenv
!git clone https://github.com/leticiavitoria/TesteTrans.git
%cd TesteTrans
!git checkout claude/setup-audio-transcription-0YKW1
```

4. Suba seu áudio (painel de arquivos à esquerda) e rode:

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
os.environ["WHISPER_DEVICE"] = "cuda"
os.environ["WHISPER_COMPUTE_TYPE"] = "float16"

!python -m src.main teste.mp3 -o saida.txt
```

5. Baixe `saida.txt` pelo painel de arquivos antes de fechar a sessão.

**Sem GPU**: troque para `WHISPER_DEVICE=cpu` e `WHISPER_COMPUTE_TYPE=int8`.
Em CPU o `large-v3` é bem mais lento (vários minutos por minuto de áudio).

## Formato de saída

```
SCENE 1
PROMPT 001 | 00:00 - 00:08 | BASE
12,000 years ago, before any human being had planted...
PROMPT 002 | 00:08 - 00:15 | EXT
first clay pot was shaped by human hands...
...

SCENE 2
PROMPT 005 | 00:25 - 00:33 | BASE
When archaeologist Klaus Schmidt...
```

## Custos

O modelo `claude-opus-4-6` é usado em duas chamadas:

- **Principal**: análise da transcrição inteira.
- **Segunda passada**: só se alguma seção entre fronteiras passar de 148s.
  Envia apenas o trecho longo.

Para um vídeo TOP-10 de ~35 minutos, custo típico fica na casa de
centavos por execução. Para baratear, troque para Sonnet 4.6
(`claude-sonnet-4-6`) editando `CLAUDE_MODEL` em `src/scene_planner.py`.

## Notas

- Se a chamada ao Claude falhar (rede, sem API key), o programa continua usando
  apenas a heurística determinística do regex + split balanceado por ponto final.
- Detecção de itens funciona em português e inglês ("10.", "Number 8",
  "décimo lugar", "tenth place").
- O `vad_filter` ativo no `transcribe.py` reduz alucinações do Whisper em
  trechos de silêncio.
