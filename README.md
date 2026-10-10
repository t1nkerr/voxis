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

## Install

```bash
git clone https://github.com/t1nkerr/voxis.git
cd voxis
./scripts/install.sh
```

For speaker diarization, use the full installer. It pins and builds FluidAudio from source,
so a working Xcode/Swift C++ toolchain is required. Before changing anything, the installer
compiles a tiny C++ probe to verify that Git, Swift, and the selected Xcode toolchain are usable.

```bash
./scripts/install.sh --with-diarize
```

The installer does not install Homebrew, ffmpeg, or Xcode automatically. It stops with the
exact missing prerequisite. It installs a normal global `voxis` command, creates
`~/.config/voxis/.env` only when absent, and never overwrites existing config.

To pin the global `voxis` to one exact commit, with dependencies constrained to that
commit's `uv.lock` (for example for scheduled jobs that must not pick up local edits):

```bash
./scripts/release.sh <commit> [--fluidaudio PATH]
```

Check readiness at any time:

```bash
voxis doctor
voxis doctor --require-diarize
```

Use `~/.config/voxis/.env` for machine-local secrets.

### What each feature needs

| Feature | Requirements |
| --- | --- |
| Base transcription and alignment | Base installer; no LLM key or context required |
| Anonymous speaker diarization (`SPEAKER_00`, `SPEAKER_01`, ...) | Full installer plus `--diarize`; no LLM key or context required |
| Reliable speaker-name mapping | Diarization and an LLM API key; `--context` with names and roles is strongly recommended |
| Transcript polish | Diarization and an LLM API key; `--context` is strongly recommended |

## Usage

```bash
voxis transcribe --audio meeting.m4a --out out/meeting
voxis transcribe --audio meeting.m4a --out out/meeting --diarize --polish \
  --context "Weekly product sync with Alice and Bob"
```

Diarization alone detects speaker changes and uses anonymous labels such as `SPEAKER_00`.
An LLM API key enables mapping those labels from transcript evidence such as self-introductions.
For reliable mapping, also pass available show notes through `--context`, including speaker
names and roles. Company or product names, acronyms, and other proper nouns in the context
also improve ASR term extraction and optional polish.

The first run downloads the ASR and alignment models (several GB) from Hugging Face;
later runs reuse the local cache.

Outputs under `--out`:

- `transcript.md`: user-facing transcript Markdown
- `run.json`: stable run receipt
- `_debug/`: only with `--debug`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full input/output contract.

## Diarization setup

`--diarize` uses the [FluidAudio](https://github.com/FluidInference/FluidAudio) Sortformer
CLI installed as `fluidaudio` on `PATH`. The supported installer pins the tested FluidAudio
version, builds it, and places it beside the global `voxis` command:

```bash
./scripts/install.sh --with-diarize
```

Verify the runtime dependency after install:

```bash
voxis doctor --require-diarize
voxis-bench run-sortformer --audio /path/to/short-audio.m4a --out /tmp/voxis-fluidaudio-smoke --model-id smoke
```

## Configuration

`~/.config/voxis/.env` holds machine-local secrets. Environment variables take precedence
over file values.

- `VOXIS_LLM_API_KEY` (or `BAILIAN_API_KEY`): key for ASR term extraction and
  `--polish`. Both default to `qwen3.8-flash`. Their shared endpoint defaults to Bailian/DashScope, with
  `VOXIS_LLM_BASE_URL`, `VOXIS_POLISH_MODEL`, and `VOXIS_ASR_TERMS_MODEL` overrides.
- `VOXIS_SPEAKER_MAP_API_KEY` (or `DEEPSEEK_API_KEY`): separate key for speaker-name
  mapping. Mapping defaults to official `deepseek-flash` at `https://api.deepseek.com`,
  with low thinking and no explicit completion-token cap. Provider defaults still apply.
  `VOXIS_SPEAKER_MAP_MODEL` and `VOXIS_SPEAKER_MAP_BASE_URL` override that route.
  The shared provider key is never reused for the speaker endpoint.
- Official DeepSeek uses JSON-object output, with structural checks performed locally;
  the shared Bailian route retains strict JSON Schema output. Mapping removes URL
  destinations from its copy of show notes while preserving visible link labels.
  Names absent verbatim from the input are retained with a diagnostic warning rather
  than automatically replaced with Unknown; structural validation remains enforced.
- `VOXIS_ASR_TERMS`: comma-separated glossary terms merged into every ASR prompt.
  Empty by default.
- `VOXIS_LLM_DISABLE_THINKING`: `true`/`false` control for sending `enable_thinking=false`
  to the shared ASR-term/polish endpoint. Defaults to on for DashScope URLs, off otherwise.
  Speaker mapping uses its separate low-thinking configuration.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## License

MIT, see [LICENSE](LICENSE).
