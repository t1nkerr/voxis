"""CLI for benchmark scorer smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from voxis.bench.parsers import load_candidate_spans_json, load_predicted_words, load_reference_words, parse_rttm
from voxis.bench.scoring import (
    PredictedWord,
    ReferenceSpan,
    assign_words_by_largest_overlap,
    assign_predicted_words,
    assign_words_to_voxis_prior,
    assign_words_to_spans,
    score_diarization_frames,
    score_word_speakers,
    span_shape_stats,
    turn_shape_stats,
)
from voxis.bench.sortformer import SortformerRunConfig, run_sortformer_candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxis-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score-word-speakers", help="score word speaker attribution")
    score_parser.add_argument("--reference-words", required=True, type=Path)
    score_parser.add_argument("--reference-rttm", required=True, type=Path)
    score_parser.add_argument("--predicted-words", required=True, type=Path)
    score_parser.add_argument("--candidate-name", default="candidate")
    score_parser.add_argument("--out", required=True, type=Path)

    spans_parser = subparsers.add_parser("score-diarization-spans", help="score diarization spans at word level")
    spans_parser.add_argument("--reference-words", required=True, type=Path)
    spans_parser.add_argument("--reference-rttm", required=True, type=Path)
    candidate_group = spans_parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument("--candidate-rttm", type=Path)
    candidate_group.add_argument("--candidate-json", type=Path)
    spans_parser.add_argument("--candidate-name", default="candidate")
    spans_parser.add_argument("--out", required=True, type=Path)

    frames_parser = subparsers.add_parser(
        "score-diarization-frames",
        help="score candidate diarization spans against reference RTTM without word timings",
    )
    frames_parser.add_argument("--reference-rttm", required=True, type=Path)
    frames_candidate_group = frames_parser.add_mutually_exclusive_group(required=True)
    frames_candidate_group.add_argument("--candidate-rttm", type=Path)
    frames_candidate_group.add_argument("--candidate-json", type=Path)
    frames_parser.add_argument("--candidate-name", default="candidate")
    frames_parser.add_argument("--frame-step-seconds", type=float, default=0.01)
    frames_parser.add_argument("--out", required=True, type=Path)

    assign_parser = subparsers.add_parser(
        "assign-diarization-words",
        help="assign reference/ASR words to candidate diarization spans",
    )
    assign_parser.add_argument("--reference-words", required=True, type=Path)
    assign_candidate_group = assign_parser.add_mutually_exclusive_group(required=True)
    assign_candidate_group.add_argument("--candidate-rttm", type=Path)
    assign_candidate_group.add_argument("--candidate-json", type=Path)
    assign_parser.add_argument("--out", required=True, type=Path)

    assign_overlap_parser = subparsers.add_parser(
        "assign-diarization-words-overlap",
        help="assign words by largest candidate-span overlap with coverage metadata",
    )
    assign_overlap_parser.add_argument("--reference-words", required=True, type=Path)
    assign_overlap_candidate_group = assign_overlap_parser.add_mutually_exclusive_group(required=True)
    assign_overlap_candidate_group.add_argument("--candidate-rttm", type=Path)
    assign_overlap_candidate_group.add_argument("--candidate-json", type=Path)
    assign_overlap_parser.add_argument("--coverage-threshold", type=float, default=0.5)
    assign_overlap_parser.add_argument("--out", required=True, type=Path)

    sortformer_parser = subparsers.add_parser(
        "run-sortformer",
        help="bench-only: run FluidAudio Sortformer and normalize candidate spans",
    )
    sortformer_parser.add_argument("--audio", required=True, type=Path)
    sortformer_parser.add_argument("--model-id", default="FluidAudio Sortformer")
    sortformer_parser.add_argument("--onset", type=float)
    sortformer_parser.add_argument("--offset", type=float)
    sortformer_parser.add_argument("--pad-onset", type=float)
    sortformer_parser.add_argument("--pad-offset", type=float)
    sortformer_parser.add_argument("--min-duration-on", type=float)
    sortformer_parser.add_argument("--min-duration-off", type=float)
    sortformer_parser.add_argument("--debug", action="store_true")
    sortformer_parser.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "score-word-speakers":
        return _score_word_speakers(args)
    if args.command == "score-diarization-spans":
        return _score_diarization_spans(args)
    if args.command == "score-diarization-frames":
        return _score_diarization_frames(args)
    if args.command == "assign-diarization-words":
        return _assign_diarization_words(args)
    if args.command == "assign-diarization-words-overlap":
        return _assign_diarization_words_overlap(args)
    if args.command == "run-sortformer":
        return _run_sortformer(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _score_word_speakers(args: argparse.Namespace) -> int:
    reference_words, dropped_words = load_reference_words(args.reference_words)
    reference_spans = parse_rttm(args.reference_rttm)
    predicted_words = load_predicted_words(args.predicted_words)
    predicted_speakers = assign_predicted_words(reference_words, predicted_words)
    score = score_word_speakers(reference_words, predicted_speakers, reference_spans)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate": args.candidate_name,
        "reference_words": str(args.reference_words),
        "reference_rttm": str(args.reference_rttm),
        "predicted_words": str(args.predicted_words),
        "dropped_reference_words": dropped_words,
        "word_speaker": score.to_json(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render_markdown_report(report), encoding="utf-8")
    return 0


def _score_diarization_spans(args: argparse.Namespace) -> int:
    reference_words, dropped_words = load_reference_words(args.reference_words)
    reference_spans = parse_rttm(args.reference_rttm)
    candidate_path, candidate_format, candidate_spans = _load_candidate_spans(args)
    predicted_speakers = assign_words_to_spans(reference_words, candidate_spans)
    voxis_prior_speakers = assign_words_to_voxis_prior(reference_words, candidate_spans)
    score = score_word_speakers(reference_words, predicted_speakers, reference_spans)
    raw_with_overlap = score_diarization_frames(reference_spans, candidate_spans, include_overlap=True)
    raw_without_overlap = score_diarization_frames(reference_spans, candidate_spans, include_overlap=False)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate": args.candidate_name,
        "reference_words": str(args.reference_words),
        "reference_rttm": str(args.reference_rttm),
        "candidate_path": str(candidate_path),
        "candidate_format": candidate_format,
        "candidate_span_count": len(candidate_spans),
        "candidate_span_stats": span_shape_stats(candidate_spans),
        "voxis_prior_turn_shape": turn_shape_stats(reference_words, voxis_prior_speakers),
        "dropped_reference_words": dropped_words,
        "word_speaker": score.to_json(),
        "raw_diarization": {
            "no_collar_include_overlap": raw_with_overlap.to_json(),
            "no_collar_exclude_overlap": raw_without_overlap.to_json(),
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render_markdown_report(report), encoding="utf-8")
    return 0


def _score_diarization_frames(args: argparse.Namespace) -> int:
    reference_spans = parse_rttm(args.reference_rttm)
    candidate_path, candidate_format, candidate_spans = _load_candidate_spans(args)
    with_overlap = score_diarization_frames(
        reference_spans,
        candidate_spans,
        frame_step_seconds=args.frame_step_seconds,
        include_overlap=True,
    )
    without_overlap = score_diarization_frames(
        reference_spans,
        candidate_spans,
        frame_step_seconds=args.frame_step_seconds,
        include_overlap=False,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate": args.candidate_name,
        "reference_rttm": str(args.reference_rttm),
        "reference_span_count": len(reference_spans),
        "candidate_path": str(candidate_path),
        "candidate_format": candidate_format,
        "candidate_span_count": len(candidate_spans),
        "candidate_span_stats": span_shape_stats(candidate_spans),
        "raw_diarization": {
            "no_collar_include_overlap": with_overlap.to_json(),
            "no_collar_exclude_overlap": without_overlap.to_json(),
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render_frame_markdown_report(report), encoding="utf-8")
    return 0


def _assign_diarization_words(args: argparse.Namespace) -> int:
    reference_words, dropped_words = load_reference_words(args.reference_words)
    candidate_path, candidate_format, candidate_spans = _load_candidate_spans(args)
    predicted_speakers = assign_words_to_voxis_prior(reference_words, candidate_spans)
    predicted_words = [
        PredictedWord(
            text=word.text,
            start=word.start,
            end=word.end,
            speaker=speaker or "UNKNOWN",
        )
        for word, speaker in zip(reference_words, predicted_speakers, strict=True)
    ]
    payload = {
        "source_reference_words": str(args.reference_words),
        "candidate_path": str(candidate_path),
        "candidate_format": candidate_format,
        "dropped_reference_words": dropped_words,
        "candidate_span_stats": span_shape_stats(candidate_spans),
        "turn_shape": turn_shape_stats(predicted_words, [word.speaker for word in predicted_words]),
        "words": [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "speaker": word.speaker,
            }
            for word in predicted_words
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _assign_diarization_words_overlap(args: argparse.Namespace) -> int:
    reference_words, dropped_words = load_reference_words(args.reference_words)
    candidate_path, candidate_format, candidate_spans = _load_candidate_spans(args)
    assignments = assign_words_by_largest_overlap(
        reference_words,
        candidate_spans,
        coverage_threshold=args.coverage_threshold,
    )
    predicted_words = [
        PredictedWord(
            text=assignment.word.text,
            start=assignment.word.start,
            end=assignment.word.end,
            speaker=assignment.assigned_speaker,
        )
        for assignment in assignments
    ]
    payload = {
        "source_reference_words": str(args.reference_words),
        "candidate_path": str(candidate_path),
        "candidate_format": candidate_format,
        "coverage_threshold": args.coverage_threshold,
        "dropped_reference_words": dropped_words,
        "candidate_span_stats": span_shape_stats(candidate_spans),
        "turn_shape": turn_shape_stats(predicted_words, [word.speaker for word in predicted_words]),
        "assignment": {
            "midpoint_disagreements": sum(
                1
                for assignment in assignments
                if assignment.midpoint_speaker is not None
                and assignment.midpoint_speaker != assignment.assigned_speaker
            ),
            "low_coverage_words": sum(
                assignment.coverage_ratio < args.coverage_threshold for assignment in assignments
            ),
        },
        "words": [
            {
                "text": assignment.word.text,
                "start": assignment.word.start,
                "end": assignment.word.end,
                "speaker": assignment.assigned_speaker,
                "midpointSpeaker": assignment.midpoint_speaker,
                "largestOverlapSpeaker": assignment.largest_overlap_speaker,
                "coverageRatio": assignment.coverage_ratio,
            }
            for assignment in assignments
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _run_sortformer(args: argparse.Namespace) -> int:
    config = SortformerRunConfig(
        audio=args.audio,
        out=args.out,
        model_id=args.model_id,
        onset=args.onset,
        offset=args.offset,
        pad_onset=args.pad_onset,
        pad_offset=args.pad_offset,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        debug=args.debug,
    )
    run_sortformer_candidate(config)
    return 0


def _load_candidate_spans(args: argparse.Namespace) -> tuple[Path, str, list[ReferenceSpan]]:
    candidate_path = args.candidate_rttm or args.candidate_json
    if candidate_path is None:
        raise AssertionError("candidate path must be provided")
    candidate_format = "rttm" if args.candidate_rttm else "json"
    candidate_spans = parse_rttm(candidate_path) if args.candidate_rttm else load_candidate_spans_json(candidate_path)
    return candidate_path, candidate_format, candidate_spans


def _render_markdown_report(report: dict[str, object]) -> str:
    word_speaker = report["word_speaker"]
    if not isinstance(word_speaker, dict):
        raise TypeError("word_speaker report must be a dict")
    main_accuracy = _format_accuracy(word_speaker.get("main_accuracy"))
    overlap_accuracy = _format_accuracy(word_speaker.get("overlap_accuracy"))
    turn_shape = report.get("voxis_prior_turn_shape")
    turn_lines: list[str] = []
    if isinstance(turn_shape, dict):
        turn_lines = [
            "",
            "## Voxis-prior turn shape",
            "",
            f"- Turns: {turn_shape.get('turns', 'n/a')}",
            f"- Turns per minute: {_format_number(turn_shape.get('turns_per_minute'))}",
            f"- Speaker switches per minute: {_format_number(turn_shape.get('speaker_switches_per_minute'))}",
            f"- Median turn: {_format_seconds(turn_shape.get('median_duration_seconds'))}",
            f"- Turns <0.5s: {turn_shape.get('under_0_5_seconds', 'n/a')}",
            f"- Turns <1.0s: {turn_shape.get('under_1_0_seconds', 'n/a')}",
        ]
    raw_diarization = report.get("raw_diarization")
    raw_lines: list[str] = []
    if isinstance(raw_diarization, dict):
        include_overlap = raw_diarization.get("no_collar_include_overlap")
        exclude_overlap = raw_diarization.get("no_collar_exclude_overlap")
        raw_lines = [
            "",
            "## Raw diarization diagnostics",
            "",
            f"- No-collar DER-like, overlap included: {_format_der(include_overlap)}",
            f"- No-collar DER-like, overlap excluded: {_format_der(exclude_overlap)}",
        ]
    return "\n".join(
        [
            f"# Benchmark report: {report['candidate']}",
            "",
            "## Word-speaker scoring",
            "",
            f"- Main accuracy: {main_accuracy}",
            f"- Main words: {word_speaker['main_correct_words']} / {word_speaker['scored_main_words']}",
            f"- Overlap accuracy: {overlap_accuracy}",
            f"- Overlap words: {word_speaker['overlap_correct_words']} / {word_speaker['overlap_words']}",
            f"- Missing predictions: {word_speaker['missing_prediction_words']}",
            f"- Dropped reference words: {report['dropped_reference_words']}",
            *turn_lines,
            *raw_lines,
            "",
        ]
    )


def _render_frame_markdown_report(report: dict[str, object]) -> str:
    raw_diarization = report["raw_diarization"]
    if not isinstance(raw_diarization, dict):
        raise TypeError("raw_diarization report must be a dict")
    include_overlap = raw_diarization.get("no_collar_include_overlap")
    exclude_overlap = raw_diarization.get("no_collar_exclude_overlap")
    span_stats = report.get("candidate_span_stats")
    if not isinstance(span_stats, dict):
        span_stats = {}
    return "\n".join(
        [
            f"# Diarization frame report: {report['candidate']}",
            "",
            "## Diagnostic DER-like scoring",
            "",
            f"- Reference spans: {report['reference_span_count']}",
            f"- Candidate spans: {report['candidate_span_count']}",
            f"- Median candidate span: {_format_seconds(span_stats.get('median_duration_seconds'))}",
            f"- Candidate spans <0.5s: {span_stats.get('under_0_5_seconds', 'n/a')}",
            f"- Candidate spans <1.0s: {span_stats.get('under_1_0_seconds', 'n/a')}",
            f"- No-collar DER-like, overlap included: {_format_der(include_overlap)}",
            f"- No-collar DER-like, overlap excluded: {_format_der(exclude_overlap)}",
            "",
        ]
    )


def _format_accuracy(value: object) -> str:
    if not isinstance(value, float):
        return "n/a"
    return f"{value:.1%}"


def _format_der(value: object) -> str:
    if not isinstance(value, dict):
        return "n/a"
    der_like = value.get("der_like")
    if not isinstance(der_like, float):
        return "n/a"
    return f"{der_like:.1%}"


def _format_seconds(value: object) -> str:
    if not isinstance(value, float | int):
        return "n/a"
    return f"{value:.2f}s"


def _format_number(value: object) -> str:
    if not isinstance(value, float | int):
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
