# Voxis Architecture

## Scope

Voxis is a standalone CLI for turning a local audio file into transcript Markdown.

## Command

```bash
voxis transcribe --audio <audio-file> --out <output-dir> \
  [--context <text>] [--language auto|en|zh] \
  [--diarize] [--polish] [--debug]
```

## Inputs

Required:

- `--audio`: local audio file.
- `--out`: directory where Voxis writes outputs.

Optional:

- `--context`: title, show notes, speaker names and roles, source name, or glossary terms. Speaker names and
  roles improve LLM mapping of anonymous diarization labels to real people.
- `--language`: ASR language hint. Default is `auto`.
- `--diarize`: enable Sortformer diarization and timestamped speaker turns with anonymous labels. Context is
  not required for diarization itself.
- `--polish`: enable LLM transcript polish. Allowed only with `--diarize`. Requires `VOXIS_LLM_API_KEY` (or `BAILIAN_API_KEY`).
- `--debug`: write debug files.

## Config

Use `~/.config/voxis/.env` for machine-local secrets. Environment variables take precedence over file values.

Required:

- None for base transcription.

Optional:

- `VOXIS_LLM_API_KEY`: used for ASR term extraction, speaker-name mapping, and `--polish`.
  `BAILIAN_API_KEY` is accepted as a fallback.
- `VOXIS_LLM_BASE_URL`, `VOXIS_POLISH_MODEL`, `VOXIS_SPEAKER_MAP_MODEL`, `VOXIS_ASR_TERMS_MODEL`:
  provider overrides. Defaults target Alibaba Bailian/DashScope. The endpoint must support OpenAI's strict
  JSON Schema response format; transport compatibility alone is insufficient for Voxis LLM features.
- `VOXIS_ASR_TERMS`: comma-separated glossary terms merged into every ASR prompt. Empty by default.
- `VOXIS_LLM_DISABLE_THINKING`: `true`/`false` override for sending `enable_thinking=false` to the LLM
  endpoint. Default is on for DashScope URLs and off otherwise.

Machine setup:

- ffmpeg must be installed for audio decoding.
- [FluidAudio](https://github.com/FluidInference/FluidAudio) CLI must be installed as `fluidaudio` on `PATH`.
  Use `./scripts/install.sh --with-diarize` to pin, build, install, and verify it.

## Providers

- ASR: local MLX Audio, `mlx-community/Qwen3-ASR-1.7B-8bit`.
  - Voxis prefixes ASR prompts with glossary terms from `VOXIS_ASR_TERMS` (empty by default), then appends
    extracted terms when context and an LLM key are available. The combined capped bare word list is passed as
    Qwen3-ASR context. If ASR output echoes prompt terms as a contiguous tail, Voxis trims that tail before
    alignment. Prompt echo outside the tail fails the ASR stage.
- Alignment: local MLX Audio, `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`.
- Diarization: FluidAudio Sortformer high-context, invoked as `fluidaudio sortformer --config high`.
- Sortformer word policy: largest-overlap assignment plus language-profiled boundary repair.
  FluidAudio receives no language setting. With `--language auto`, Voxis reads ASR segment language metadata
  and chooses the English or Chinese post-assignment policy profile. Explicit `--language en|zh` overrides
  ASR-detected language for both ASR and policy routing.
- Speaker mapping: JSON schema mode on code default model `deepseek-v4-flash`.
- Polish: OpenAI-compatible chat endpoint on code default model `deepseek-v4-flash`.
  - Default base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1` (override with `VOXIS_LLM_BASE_URL`).
  - `enable_thinking=false` is sent only to DashScope endpoints unless `VOXIS_LLM_DISABLE_THINKING` overrides it.
  - Polish chunks run with bounded concurrency using the code default worker cap.

Record provider names and model/profile IDs in `run.json`.

## Outputs

Required:

```text
<output-dir>/
  transcript.md
  run.json
```

`transcript.md` is the user-facing transcript Markdown.
`run.json` is the stable receipt for the run.

Transcript shape:

```md
# Transcript

<plain ASR transcript paragraphs>
```

When `--diarize` is used:

```md
# Transcript

[00:00:12] Speaker 1: <speaker turn>

[00:00:34] Speaker 2: <speaker turn>
```

When `--polish` is used, polish receives chunked speaker turns with internal timestamp ranges:

```md
[00:00:12.30 - 00:00:18.90] Speaker 1: <speaker turn>
```

Polish returns exactly one polished text string per input turn, in the same order. It may remove
non-substantive filler/false starts/repetitions, fix punctuation and spacing, and correct ASR
spelling/terminology mistakes supported by `--context`. It must not merge, split, drop, reorder, or
reassign turns, add speakers, summarize, translate, or add headings; turn count, order, timestamps, and
speaker labels are preserved exactly, and the validator hard-fails missing or extra turn keys.
The final `transcript.md` still renders user-facing timestamps as `[HH:MM:SS]`.

Polish runs in chunks. Hard-failed chunks fall back to original timestamped turns; warning-only chunks are kept and logged.

Optional, only when `--debug` is used:

```text
<output-dir>/
  transcript.md
  run.json
  _debug/
    progress.jsonl       # stage progress, written during the run
    asr_segments.json
    terms.txt            # capped ASR spelling terms passed to Qwen3-ASR, if any
    diarization.json      # only when --diarize is used
    speaker_map.json      # only when speaker mapping is attempted
    polish.json           # only when --polish is used
```

Debug files are inspect-only and may change. Callers should depend on `transcript.md` and `run.json`.

## Boundaries

- Base transcription should run without Hugging Face or LLM config.
- Voxis v1 accepts local audio files only. Callers must download remote audio before invoking Voxis.
- Voxis writes only under `--out`.
