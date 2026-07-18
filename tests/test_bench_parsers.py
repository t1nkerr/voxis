from pathlib import Path

import pytest

from voxis.bench.parsers import (
    filter_reference_words,
    load_candidate_spans_json,
    load_reference_words,
    parse_ami_segments_xml,
    parse_ami_words_xml,
    parse_rttm,
)
from voxis.bench.scoring import ReferenceWord


def test_filter_reference_words_drops_annotation_artifacts_but_keeps_real_fillers() -> None:
    words = [
        ReferenceWord("[noise]", 0.0, 0.1, "A"),
        ReferenceWord("<laugh>", 0.1, 0.2, "A"),
        ReferenceWord("um", 0.2, 0.3, "A"),
        ReferenceWord("word", 0.3, 0.4, "A"),
    ]

    kept, dropped = filter_reference_words(words)

    assert [word.text for word in kept] == ["um", "word"]
    assert dropped == 2


def test_load_reference_words_accepts_object_with_words_key(tmp_path: Path) -> None:
    path = tmp_path / "reference_words.json"
    path.write_text(
        """
        {
          "words": [
            {"text": "hello", "start": 0.0, "end": 0.4, "speaker": "A"},
            {"text": "[noise]", "start": 0.4, "end": 0.5, "speaker": "A"},
            {"text": ".", "start": 0.5, "end": 0.5, "speaker": "A"}
          ]
        }
        """,
        encoding="utf-8",
    )

    words, dropped = load_reference_words(path)

    assert [word.text for word in words] == ["hello"]
    assert dropped == 2


def test_parse_rttm_reads_speaker_spans(tmp_path: Path) -> None:
    path = tmp_path / "reference.rttm"
    path.write_text(
        "SPEAKER sample 1 1.500 2.250 <NA> <NA> Alice <NA> <NA>\n",
        encoding="utf-8",
    )

    spans = parse_rttm(path)

    assert len(spans) == 1
    assert spans[0].start == 1.5
    assert spans[0].end == 3.75
    assert spans[0].speaker == "Alice"


def test_parse_rttm_rejects_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "reference.rttm"
    path.write_text("BAD sample\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected RTTM"):
        parse_rttm(path)


def test_load_candidate_spans_json_accepts_fluidaudio_shape(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(
        """
        {
          "segments": [
            {
              "startTimeSeconds": 1.25,
              "endTimeSeconds": 2.50,
              "speakerId": "S1"
            },
            {
              "start": 3.00,
              "end": 4.00,
              "speakerIndex": 2
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    spans = load_candidate_spans_json(path)

    assert [(span.start, span.end, span.speaker) for span in spans] == [
        (1.25, 2.5, "S1"),
        (3.0, 4.0, "2"),
    ]


def test_parse_ami_words_xml_reads_timings_and_speaker_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "ES2002a.A.words.xml"
    path.write_text(
        """
        <nite:root xmlns:nite="http://nite.sourceforge.net/">
          <w starttime="0.10" endtime="0.20">hello</w>
          <w starttime="0.20" endtime="0.30">[noise]</w>
          <w starttime="0.30" endtime="0.30" punc="true">.</w>
        </nite:root>
        """,
        encoding="utf-8",
    )

    words, dropped = parse_ami_words_xml(path)

    assert [(word.text, word.start, word.end, word.speaker) for word in words] == [("hello", 0.1, 0.2, "A")]
    assert dropped == 2


def test_parse_ami_words_xml_drops_untimed_words(tmp_path: Path) -> None:
    path = tmp_path / "ES2008a.A.words.xml"
    path.write_text(
        """
        <nite:root xmlns:nite="http://nite.sourceforge.net/">
          <w starttime="0.10" endtime="0.20">timed</w>
          <w>untimed</w>
          <w punc="true">.</w>
        </nite:root>
        """,
        encoding="utf-8",
    )

    words, dropped = parse_ami_words_xml(path)

    assert [word.text for word in words] == ["timed"]
    assert dropped == 2


def test_parse_ami_words_xml_prefers_explicit_speaker(tmp_path: Path) -> None:
    path = tmp_path / "dummy.words.xml"
    path.write_text(
        """
        <root>
          <w start="0.10" end="0.20" speaker="SpeakerA">hello</w>
        </root>
        """,
        encoding="utf-8",
    )
    words, dropped = parse_ami_words_xml(path, speaker="fallback")

    assert dropped == 0
    assert words[0].speaker == "SpeakerA"


def test_parse_ami_segments_xml_reads_segment_timings_and_speaker(tmp_path: Path) -> None:
    path = tmp_path / "ES2002a.C.segments.xml"
    path.write_text(
        """
        <root>
          <segment transcriber_start="10.5" transcriber_end="12.0" />
        </root>
        """,
        encoding="utf-8",
    )

    spans = parse_ami_segments_xml(path)

    assert len(spans) == 1
    assert spans[0].start == 10.5
    assert spans[0].end == 12.0
    assert spans[0].speaker == "C"
