from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from voxis import __version__
from voxis.config import load_config
from voxis.doctor import run_doctor
from voxis.run import TranscribeOptions, transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxis", description="Standalone audio transcription CLI.")
    parser.add_argument("--version", action="version", version=f"voxis {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe a local audio file.")
    transcribe_parser.add_argument("--audio", required=True, help="Local audio file to transcribe.")
    transcribe_parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    transcribe_parser.add_argument("--context", help="Show notes for terminology, speaker-name mapping, and polish.")
    transcribe_parser.add_argument(
        "--language", choices=("auto", "en", "zh"), default="auto", help="ASR language hint."
    )
    transcribe_parser.add_argument(
        "--diarize",
        action="store_true",
        help="Label speaker turns; an LLM key enables names and context improves mapping.",
    )
    transcribe_parser.add_argument("--polish", action="store_true", help="Polish diarized transcript with an LLM.")
    transcribe_parser.add_argument("--debug", action="store_true", help="Write debug files under _debug/.")
    transcribe_parser.set_defaults(func=_run_transcribe)

    doctor_parser = subparsers.add_parser("doctor", help="Check local Voxis dependencies.")
    doctor_parser.add_argument(
        "--require-diarize",
        action="store_true",
        help="Return an error when the optional FluidAudio CLI is unavailable.",
    )
    doctor_parser.set_defaults(func=_run_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args, parser))


def _run_transcribe(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.polish and not args.diarize:
        parser.error("--polish is allowed only with --diarize")
    raw_audio = str(args.audio)
    if raw_audio.startswith(("http://", "https://")):
        parser.error("--audio must be a local file path")
    audio_path = Path(raw_audio).expanduser()
    if not audio_path.exists():
        parser.error(f"--audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        parser.error(f"--audio must point to a file: {audio_path}")
    out_path = args.out.expanduser()
    if out_path.exists() and not out_path.is_dir():
        parser.error(f"--out must be a directory path: {out_path}")

    config = load_config()
    if args.polish and config.llm_api_key is None:
        parser.error("--polish requires VOXIS_LLM_API_KEY (or BAILIAN_API_KEY) in ~/.config/voxis/.env or environment")

    options = TranscribeOptions(
        audio=audio_path,
        out=out_path,
        context=args.context,
        language=args.language,
        diarize=args.diarize,
        polish=args.polish,
        debug=args.debug,
    )
    try:
        return transcribe(options, config)
    except Exception as exc:
        print(f"voxis: error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


def _run_doctor(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    del parser
    return run_doctor(require_diarize=bool(args.require_diarize))
