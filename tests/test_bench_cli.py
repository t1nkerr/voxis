import json
import sys
from pathlib import Path

from pytest import MonkeyPatch

from voxis.bench.cli import main


def test_score_word_speakers_cli_writes_json_and_markdown_report(tmp_path: Path) -> None:
    reference_words = tmp_path / "reference_words.json"
    reference_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Alice"},
                {"text": "reply", "start": 1.0, "end": 2.0, "speaker": "Bob"},
            ]
        ),
        encoding="utf-8",
    )
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> Alice <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> Bob <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predicted_words = tmp_path / "predicted_words.json"
    predicted_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
                {"text": "reply", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = main(
        [
            "score-word-speakers",
            "--reference-words",
            str(reference_words),
            "--reference-rttm",
            str(reference_rttm),
            "--predicted-words",
            str(predicted_words),
            "--candidate-name",
            "candidate_auto",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["candidate"] == "candidate_auto"
    assert report["word_speaker"]["main_accuracy"] == 1.0
    assert report["word_speaker"]["mapping"] == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert "Main accuracy: 100.0%" in (out / "report.md").read_text(encoding="utf-8")


def test_score_diarization_spans_cli_scores_candidate_rttm(tmp_path: Path) -> None:
    reference_words = tmp_path / "reference_words.json"
    reference_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Alice"},
                {"text": "reply", "start": 1.0, "end": 2.0, "speaker": "Bob"},
            ]
        ),
        encoding="utf-8",
    )
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> Alice <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> Bob <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_rttm = tmp_path / "candidate.rttm"
    candidate_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = main(
        [
            "score-diarization-spans",
            "--reference-words",
            str(reference_words),
            "--reference-rttm",
            str(reference_rttm),
            "--candidate-rttm",
            str(candidate_rttm),
            "--candidate-name",
            "candidate_auto",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["candidate"] == "candidate_auto"
    assert report["candidate_span_count"] == 2
    assert report["candidate_format"] == "rttm"
    assert report["word_speaker"]["main_accuracy"] == 1.0
    assert report["word_speaker"]["mapping"] == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert report["raw_diarization"]["no_collar_include_overlap"]["der_like"] == 0.0
    assert report["voxis_prior_turn_shape"] == {
        "turns": 2,
        "turns_per_minute": 60.0,
        "speaker_switches_per_minute": 30.0,
        "median_duration_seconds": 1.0,
        "under_0_5_seconds": 0,
        "under_1_0_seconds": 0,
    }


def test_score_diarization_spans_cli_scores_candidate_json(tmp_path: Path) -> None:
    reference_words = tmp_path / "reference_words.json"
    reference_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Alice"},
                {"text": "reply", "start": 1.0, "end": 2.0, "speaker": "Bob"},
            ]
        ),
        encoding="utf-8",
    )
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> Alice <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> Bob <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"startTimeSeconds": 0.0, "endTimeSeconds": 1.0, "speakerId": "SPEAKER_01"},
                    {"startTimeSeconds": 1.0, "endTimeSeconds": 2.0, "speakerId": "SPEAKER_00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = main(
        [
            "score-diarization-spans",
            "--reference-words",
            str(reference_words),
            "--reference-rttm",
            str(reference_rttm),
            "--candidate-json",
            str(candidate_json),
            "--candidate-name",
            "fluid_offline",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["candidate"] == "fluid_offline"
    assert report["candidate_format"] == "json"
    assert report["candidate_span_count"] == 2
    assert report["word_speaker"]["main_accuracy"] == 1.0
    assert report["word_speaker"]["mapping"] == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}
    assert report["raw_diarization"]["no_collar_exclude_overlap"]["der_like"] == 0.0


def test_score_diarization_frames_cli_scores_rttm_without_reference_words(tmp_path: Path) -> None:
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> Alice <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> Bob <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_rttm = tmp_path / "candidate.rttm"
    candidate_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = main(
        [
            "score-diarization-frames",
            "--reference-rttm",
            str(reference_rttm),
            "--candidate-rttm",
            str(candidate_rttm),
            "--candidate-name",
            "candidate_auto",
            "--frame-step-seconds",
            "0.5",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["candidate"] == "candidate_auto"
    assert report["reference_span_count"] == 2
    assert report["candidate_span_count"] == 2
    assert report["candidate_format"] == "rttm"
    assert report["raw_diarization"]["no_collar_include_overlap"]["der_like"] == 0.0
    assert "Diarization frame report" in (out / "report.md").read_text(encoding="utf-8")


def test_score_diarization_frames_cli_scores_json_without_reference_words(tmp_path: Path) -> None:
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> Alice <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> Bob <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"startTimeSeconds": 0.0, "endTimeSeconds": 1.0, "speakerId": "SPEAKER_01"},
                    {"startTimeSeconds": 1.0, "endTimeSeconds": 2.0, "speakerId": "SPEAKER_00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = main(
        [
            "score-diarization-frames",
            "--reference-rttm",
            str(reference_rttm),
            "--candidate-json",
            str(candidate_json),
            "--candidate-name",
            "fluid_offline",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["candidate"] == "fluid_offline"
    assert report["candidate_format"] == "json"
    assert report["candidate_span_stats"] == {
        "segments": 2,
        "median_duration_seconds": 1.0,
        "under_0_5_seconds": 0,
        "under_1_0_seconds": 0,
    }
    assert report["raw_diarization"]["no_collar_exclude_overlap"]["der_like"] == 0.0


def test_assign_diarization_words_cli_writes_assigned_words(tmp_path: Path) -> None:
    reference_words = tmp_path / "reference_words.json"
    reference_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Alice"},
                {"text": "reply", "start": 1.0, "end": 2.0, "speaker": "Bob"},
            ]
        ),
        encoding="utf-8",
    )
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"startTimeSeconds": 0.0, "endTimeSeconds": 1.0, "speakerId": "S1"},
                    {"startTimeSeconds": 1.0, "endTimeSeconds": 2.0, "speakerId": "S2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "assigned_words.json"

    exit_code = main(
        [
            "assign-diarization-words",
            "--reference-words",
            str(reference_words),
            "--candidate-json",
            str(candidate_json),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidate_format"] == "json"
    assert payload["candidate_span_stats"]["segments"] == 2
    assert payload["turn_shape"]["turns"] == 2
    assert payload["words"] == [
        {"end": 1.0, "speaker": "S1", "start": 0.0, "text": "hello"},
        {"end": 2.0, "speaker": "S2", "start": 1.0, "text": "reply"},
    ]


def test_assign_diarization_words_overlap_cli_writes_coverage_metadata(tmp_path: Path) -> None:
    reference_words = tmp_path / "reference_words.json"
    reference_words.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "Alice"},
            ]
        ),
        encoding="utf-8",
    )
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"startTimeSeconds": 0.0, "endTimeSeconds": 0.4, "speakerId": "S1"},
                    {"startTimeSeconds": 0.4, "endTimeSeconds": 1.0, "speakerId": "S2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "assigned_words.json"

    exit_code = main(
        [
            "assign-diarization-words-overlap",
            "--reference-words",
            str(reference_words),
            "--candidate-json",
            str(candidate_json),
            "--coverage-threshold",
            "0.5",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["assignment"] == {"low_coverage_words": 0, "midpoint_disagreements": 0}
    assert payload["words"] == [
        {
            "coverageRatio": 0.6,
            "end": 1.0,
            "largestOverlapSpeaker": "S2",
            "midpointSpeaker": "S2",
            "speaker": "S2",
            "start": 0.0,
            "text": "hello",
        }
    ]


def test_run_sortformer_cli_normalizes_candidate_and_metadata(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "fluidaudio"
    executable.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json",
                "import sys",
                "from pathlib import Path",
                "assert sys.argv[1] == 'sortformer'",
                "assert '--onset' in sys.argv",
                "assert '--min-duration-off' in sys.argv",
                "output = Path(sys.argv[sys.argv.index('--output') + 1])",
                "output.write_text(json.dumps({'segments': [",
                "    {'startTimeSeconds': 0.0, 'endTimeSeconds': 1.0, 'speaker': 'Speaker 0'},",
                "    {'startTimeSeconds': 1.2, 'endTimeSeconds': 2.0, 'speaker': 'Speaker 0'},",
                "    {'startTimeSeconds': 3.0, 'endTimeSeconds': 3.1, 'speaker': 'Speaker 1'},",
                "]}), encoding='utf-8')",
                "print('mock ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    out = tmp_path / "sortformer"

    exit_code = main(
        [
            "run-sortformer",
            "--audio",
            str(audio),
            "--model-id",
            "sortformer-test",
            "--onset",
            "0.64",
            "--min-duration-off",
            "0.15",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    candidate = json.loads((out / "candidate.json").read_text(encoding="utf-8"))
    assert candidate["segments"] == [
        {"endTimeSeconds": 1.0, "speakerId": "Speaker 0", "startTimeSeconds": 0.0},
        {"endTimeSeconds": 2.0, "speakerId": "Speaker 0", "startTimeSeconds": 1.2},
        {"endTimeSeconds": 3.1, "speakerId": "Speaker 1", "startTimeSeconds": 3.0},
    ]
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "sortformer"
    assert metadata["model_id"] == "sortformer-test"
    assert metadata["post_processing"]["sortformer_native"]["onset"] == 0.64
    assert metadata["post_processing"]["sortformer_native"]["min_duration_off"] == 0.15
    assert metadata["raw_span_stats"]["segments"] == 3
    assert metadata["candidate_span_stats"]["segments"] == 3
    assert (out / "sortformer.stdout.log").read_text(encoding="utf-8").strip() == "mock ok"
