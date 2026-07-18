# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-18

Initial public release.

- Local MLX-based transcription (`voxis transcribe`) with Qwen3-ASR and forced alignment
- Optional speaker diarization via the FluidAudio Sortformer CLI (`--diarize`)
- Optional LLM term extraction, speaker-name mapping, and transcript polish (`--polish`)
- OpenAI-compatible LLM endpoint configuration via `VOXIS_LLM_*` environment variables
  (defaults to Alibaba Bailian/DashScope; `BAILIAN_API_KEY` also accepted)
- `transcript.md` and `run.json` outputs, `_debug/` diagnostics with `--debug`
