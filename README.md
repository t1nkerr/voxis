# Voxis

Standalone audio transcription CLI for Apple Silicon Macs. Local MLX-based ASR and forced
alignment, optional speaker diarization and LLM transcript polish.

## Requirements

- macOS on Apple Silicon (MLX)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- ffmpeg for audio decoding (`brew install ffmpeg`)
- Optional, for `--diarize`: the [FluidAudio](https://github.com/FluidInference/FluidAudio)
  CLI installed as `fluidaudio` on `PATH` (see below)
- Optional, for LLM features: an API key for an OpenAI-compatible endpoint
  (defaults to Alibaba Bailian/DashScope)

## Setup

```bash
uv sync
mkdir -p ~/.config/voxis
cp .env.example ~/.config/voxis/.env
```

Use `~/.config/voxis/.env` for machine-local secrets. A repo-local `.env` is only for
temporary development overrides.

## Usage

```bash
uv run voxis transcribe --audio meeting.m4a --out out/meeting
uv run voxis transcribe --audio meeting.m4a --out out/meeting --diarize --polish \
  --context "Weekly product sync with Alice and Bob"
```

The first run downloads the ASR and alignment models (several GB) from Hugging Face;
later runs reuse the local cache.

Outputs under `--out`:

- `transcript.md`: user-facing transcript Markdown
- `run.json`: stable run receipt
- `_debug/`: only with `--debug`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full input/output contract.

## Diarization setup

`--diarize` uses the [FluidAudio](https://github.com/FluidInference/FluidAudio) Sortformer
CLI installed as `fluidaudio` on `PATH`. Install it from a FluidAudio checkout with:

```bash
scripts/install_fluidaudio_cli.sh /path/to/FluidAudio
```

Verify the runtime dependency after install:

```bash
uv run which fluidaudio
uv run voxis-bench run-sortformer --audio /path/to/short-audio.m4a --out /tmp/voxis-fluidaudio-smoke --model-id smoke
```

## Configuration

`~/.config/voxis/.env` holds machine-local secrets. Environment variables take precedence
over file values.

- `VOXIS_LLM_API_KEY`: API key for the OpenAI-compatible LLM endpoint used for ASR term
  extraction, speaker-name mapping, and `--polish`. `BAILIAN_API_KEY` is also accepted.
- `VOXIS_LLM_BASE_URL`, `VOXIS_POLISH_MODEL`, `VOXIS_SPEAKER_MAP_MODEL`,
  `VOXIS_ASR_TERMS_MODEL`: optional provider overrides. Defaults target Alibaba
  Bailian/DashScope.
- `VOXIS_ASR_TERMS`: comma-separated glossary terms merged into every ASR prompt.
  Empty by default.
- `VOXIS_LLM_DISABLE_THINKING`: `true`/`false` control for sending `enable_thinking=false`
  to the LLM endpoint. Defaults to on for DashScope URLs, off otherwise.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## License

MIT, see [LICENSE](LICENSE).
