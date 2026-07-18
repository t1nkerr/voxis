from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
from typing import Any

from voxis.config import DEFAULT_POLISH_MAX_WORKERS, VoxisConfig
from voxis.diarization_types import Turn, format_timestamp
from voxis.llm import call_json_schema_llm, no_thinking_extra_body

POLISH_PROMPT_TEMPLATE_ID = "keyed_text_exact_turn_polish_chunk_v2"
POLISH_CHUNK_TARGET_CHARS = 8500
POLISH_COLLAPSE_WARNING_RATIO = 0.6
POLISH_BLOAT_WARNING_RATIO = 1.6
POLISH_COLLAPSE_HARD_RATIO = 0.45
POLISH_BLOAT_HARD_RATIO = 2.2
POLISH_RATIO_MIN_SOURCE_CHARS = 80
_TURN_RE = re.compile(
    r"^\[(?P<start>\d+:\d{2}:\d{2}\.\d{2}) - (?P<end>\d+:\d{2}:\d{2}\.\d{2})\] "
    r"(?P<speaker>[^:]+): (?P<text>.+)$"
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
POLISH_PROMPT = """Polish this transcript chunk.

<goals>
1. Improve readability without changing meaning.
- Fix punctuation, spacing, and non-substantive filler/false starts/repetitions.
- Do not summarize, translate, add headings, add notes, or add speaker names.

2. Correct obvious ASR spelling and terminology mistakes using <context>.
- Use <context> to fix misspelled names, products, companies, models, acronyms, and technical terms.
- Prefer corrections supported by <context>.
- If <context> does not contain the term, correct only obvious common-word ASR errors where <transcript> makes the correction unambiguous.
- Do not guess names, products, companies, models, acronyms, or technical terms not supported by <context>.

3. Preserve transcript structure exactly.
- Return exactly one polished text string for each input turn, in the same order.
- Do not include timestamps, speaker labels, headings, notes, or commentary in the output strings.
- Do not merge, split, drop, reorder, or reassign turns.
- Only edit the spoken text after the speaker label.
</goals>

<output_format>
Return exactly one JSON object and nothing else.
- The response must start with {{ and end immediately after the matching final }}.
- Do not wrap the JSON in Markdown or code fences.
- Do not repeat the JSON object.
- Do not add explanations, comments, notes, transcript text, or any trailing text after the final }}.
- Use this exact shape:
{{"texts":{{"0":"polished text for input turn 0","1":"polished text for input turn 1"}}}}
</output_format>

<context>
{context}
</context>

<transcript_json>
{turns}
</transcript_json>
"""


@dataclass(frozen=True)
class PolishResult:
    text: str
    prompt_template_id: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    status: str = "succeeded"
    issues: tuple[str, ...] = ()
    chunk_count: int = 0
    max_workers: int = DEFAULT_POLISH_MAX_WORKERS
    effective_workers: int = 1
    chunks: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ParsedTurn:
    start: float
    end: float
    speaker: str
    text: str
    line: str


def polish_turns(turns: list[Turn], *, context: str | None, config: VoxisConfig) -> PolishResult:
    chunks = _chunk_turns(turns)
    max_workers = max(1, int(config.polish_max_workers))
    effective_workers = min(max_workers, len(chunks)) if chunks else 1
    chunk_payloads = _polish_chunks(chunks, context=context, config=config, max_workers=effective_workers)
    polished_chunks: list[str] = [result["text"] for result in chunk_payloads]
    all_issues: list[dict[str, Any]] = []
    prompt_tokens = sum(int(result["prompt_tokens"]) for result in chunk_payloads)
    completion_tokens = sum(int(result["completion_tokens"]) for result in chunk_payloads)
    elapsed_seconds = sum(float(result["elapsed_seconds"]) for result in chunk_payloads)
    for index, result in enumerate(chunk_payloads):
        all_issues.extend({**issue, "chunk_index": index} for issue in result["issues"])

    fallback_used = any(result["status"] in {"failed_validation", "llm_error"} for result in chunk_payloads)
    status = "succeeded_with_fallback" if fallback_used else ("succeeded_with_warnings" if all_issues else "succeeded")
    return PolishResult(
        text="\n\n".join(chunk.strip() for chunk in polished_chunks if chunk.strip()),
        prompt_template_id=POLISH_PROMPT_TEMPLATE_ID,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=elapsed_seconds,
        status=status,
        issues=tuple(_issue_label(issue) for issue in all_issues),
        chunk_count=len(chunks),
        max_workers=max_workers,
        effective_workers=effective_workers,
        chunks=tuple(chunk_payloads),
    )


def _polish_chunks(
    chunks: list[list[Turn]], *, context: str | None, config: VoxisConfig, max_workers: int
) -> list[dict[str, Any]]:
    if max_workers <= 1 or len(chunks) <= 1:
        return [_polish_chunk(index, chunk, context=context, config=config) for index, chunk in enumerate(chunks)]

    results: list[dict[str, Any] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_polish_chunk, index, chunk, context=context, config=config): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    ordered_results: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            raise RuntimeError("missing polish chunk result")
        ordered_results.append(result)
    return ordered_results


def _polish_chunk(index: int, turns: list[Turn], *, context: str | None, config: VoxisConfig) -> dict[str, Any]:
    prompt = POLISH_PROMPT.format(
        context=(context or "").strip() or "(none)",
        turns=_turns_json(turns),
    )
    fallback = _turns_text(turns)
    try:
        response = call_json_schema_llm(
            prompt,
            config=config,
            model=config.polish_model,
            schema_name="polished_turns",
            schema=_polish_response_schema(len(turns)),
            max_tokens=_polish_max_tokens(turns),
            extra_body=no_thinking_extra_body(config),
        )
    except Exception as exc:
        return _chunk_payload(
            index=index,
            turns=turns,
            status="llm_error",
            text=fallback,
            issues=(
                _issue(
                    "LLM_ERROR",
                    "error",
                    f"LLM call failed; using original timestamped chunk: {type(exc).__name__}: {exc}",
                ),
            ),
        )

    normalized, issues = _validate_polish_response(response.text, turns)
    hard_failed = any(issue["severity"] == "error" for issue in issues)
    return _chunk_payload(
        index=index,
        turns=turns,
        status="failed_validation" if hard_failed else ("succeeded_with_warnings" if issues else "succeeded"),
        text=fallback if hard_failed else normalized,
        issues=tuple(issues),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        elapsed_seconds=response.elapsed_seconds,
    )


def _chunk_payload(
    *,
    index: int,
    turns: list[Turn],
    status: str,
    text: str,
    issues: tuple[dict[str, Any], ...],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "index": index,
        "status": status,
        "turn_count": len(turns),
        "start": turns[0].start if turns else None,
        "end": turns[-1].end if turns else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "issues": [dict(issue) for issue in issues],
        "text": text,
    }


def _chunk_turns(turns: list[Turn]) -> list[list[Turn]]:
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    current_chars = 0
    for turn in turns:
        line_chars = len(_turn_line(turn)) + 1
        if current and current_chars + line_chars > POLISH_CHUNK_TARGET_CHARS:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += line_chars
    if current:
        chunks.append(current)
    return chunks


def _turns_text(turns: list[Turn]) -> str:
    return "\n".join(_turn_line(turn) for turn in turns)


def _turns_json(turns: list[Turn]) -> str:
    payload = {
        "turns": [
            {
                "id": index,
                "speaker": turn.speaker,
                "start": _format_precise_timestamp(turn.start),
                "end": _format_precise_timestamp(turn.end),
                "text": turn.text,
            }
            for index, turn in enumerate(turns)
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _turn_line(turn: Turn) -> str:
    return (
        f"[{_format_precise_timestamp(turn.start)} - {_format_precise_timestamp(turn.end)}] {turn.speaker}: {turn.text}"
    )


def _polish_response_schema(turn_count: int) -> dict[str, Any]:
    keys = [str(index) for index in range(turn_count)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["texts"],
        "properties": {
            "texts": {
                "type": "object",
                "additionalProperties": False,
                "required": keys,
                "properties": {key: {"type": "string"} for key in keys},
            }
        },
    }


def _polish_max_tokens(turns: list[Turn]) -> int:
    source_chars = _compact_char_count(_turns_text(turns))
    return max(1024, min(16_384, int(source_chars * 1.5) + 512))


def render_user_facing_polished_text(text: str) -> str:
    turns, issues = _parse_polished_turns(text)
    if issues:
        return text.strip()
    return "\n\n".join(f"[{format_timestamp(turn.start)}] {turn.speaker}: {turn.text}" for turn in turns)


def _validate_polish_response(text: str, turns: list[Turn]) -> tuple[str, list[dict[str, Any]]]:
    raw = _strip_code_fence(text).strip()
    issues: list[dict[str, Any]] = []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return "", [
            _issue(
                "MALFORMED_JSON",
                "error",
                "polish response is not valid JSON",
                {"message": str(exc), "response_excerpt": raw[:500]},
            )
        ]
    if not isinstance(payload, dict):
        return "", [_issue("BAD_SHAPE", "error", "polish response must be a JSON object")]
    response_texts = payload.get("texts")
    if set(payload) != {"texts"}:
        issues.append(
            _issue(
                "BAD_SHAPE",
                "error",
                "polish response must contain only the texts field",
                {"keys": sorted(str(key) for key in payload)},
            )
        )
    if not isinstance(response_texts, dict):
        return "", issues + [_issue("BAD_SHAPE", "error", "polish response texts field must be an object")]

    expected_keys = {str(index) for index in range(len(turns))}
    actual_keys = {str(key) for key in response_texts}
    missing_keys = sorted(expected_keys - actual_keys, key=int)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        issues.append(
            _issue(
                "MISSING_TURN_KEY",
                "error",
                "polish response omitted input turn ids",
                {"missing": missing_keys},
            )
        )
    if extra_keys:
        issues.append(
            _issue(
                "EXTRA_TURN_KEY",
                "error",
                "polish response included unknown turn ids",
                {"extra": extra_keys},
            )
        )

    polished_turns: list[ParsedTurn] = []
    for index, turn in enumerate(turns):
        key = str(index)
        value = response_texts.get(key, "")
        if not isinstance(value, str):
            issues.append(
                _issue(
                    "BAD_TEXT_VALUE",
                    "error",
                    "polish response text item must be a string",
                    {"key": key, "value_type": type(value).__name__},
                )
            )
            value = turn.text
        cleaned = _normalize_polished_text_value(value)
        if not cleaned:
            issues.append(_issue("EMPTY_TURN_TEXT", "error", "polish response text item is empty", {"key": key}))
            cleaned = turn.text
        if "\n" in value or "\r" in value:
            issues.append(
                _issue(
                    "MULTILINE_TEXT_VALUE",
                    "error",
                    "polish response text item contains a newline",
                    {"key": key},
                )
            )
        if _HEADING_RE.match(cleaned):
            issues.append(
                _issue("MARKDOWN_HEADING", "error", "polish response text item is a markdown heading", {"key": key})
            )
        if _contains_structural_text(cleaned, allowed_speakers={turn.speaker}):
            issues.append(
                _issue(
                    "STRUCTURAL_TEXT_LEAK",
                    "error",
                    "polish response text item includes timestamps or speaker labels",
                    {"key": key, "value": cleaned[:200]},
                )
            )
        polished_turns.append(
            ParsedTurn(
                turn.start,
                turn.end,
                turn.speaker,
                cleaned,
                _parsed_turn_line(ParsedTurn(turn.start, turn.end, turn.speaker, cleaned, "")),
            )
        )

    issues.extend(_content_ratio_issues(turns, polished_turns))
    if any(issue["severity"] == "error" for issue in issues):
        return "", issues
    return "\n\n".join(_parsed_turn_line(turn) for turn in polished_turns), issues


def _normalize_polished_text_value(value: str) -> str:
    return " ".join(value.strip().split())


def _contains_structural_text(value: str, *, allowed_speakers: set[str]) -> bool:
    if re.search(r"\[\d+:\d{2}:\d{2}(?:\.\d{1,2})?\s*-", value):
        return True
    structural_prefixes = set(allowed_speakers)
    structural_prefixes.update(
        speaker for speaker in allowed_speakers if re.fullmatch(r"Unknown \(SPEAKER_\d+\)", speaker)
    )
    structural_prefixes.update(re.findall(r"SPEAKER_\d+", " ".join(allowed_speakers)))
    return any(value.startswith(f"{speaker}: ") for speaker in structural_prefixes)


def _parse_polished_turns(text: str) -> tuple[list[ParsedTurn], list[dict[str, Any]]]:
    turns: list[ParsedTurn] = []
    issues: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            issues.append(
                _issue("MARKDOWN_HEADING", "error", "polish output included a markdown heading", {"line": line_number})
            )
            continue
        match = _TURN_RE.fullmatch(line)
        if match is None:
            issues.append(
                _issue(
                    "MALFORMED_TURN_LINE",
                    "error",
                    "non-empty output line is not a transcript turn",
                    {"line": line_number, "value": raw_line},
                )
            )
            continue
        try:
            start = _parse_precise_timestamp(match.group("start"))
            end = _parse_precise_timestamp(match.group("end"))
        except ValueError as exc:
            issues.append(
                _issue(
                    "BAD_TIMESTAMP",
                    "error",
                    "turn timestamp could not be parsed",
                    {"line": line_number, "value": str(exc)},
                )
            )
            continue
        turns.append(
            ParsedTurn(
                start=start,
                end=end,
                speaker=match.group("speaker").strip(),
                text=match.group("text").strip(),
                line=line,
            )
        )
    return turns, issues


def _content_ratio_issues(input_turns: list[Turn], output_turns: list[ParsedTurn]) -> list[dict[str, Any]]:
    source_chars = _compact_char_count(" ".join(turn.text for turn in input_turns))
    output_chars = _compact_char_count(" ".join(turn.text for turn in output_turns))
    if source_chars < POLISH_RATIO_MIN_SOURCE_CHARS:
        return []
    ratio = output_chars / source_chars if source_chars else 0.0
    if ratio < POLISH_COLLAPSE_HARD_RATIO:
        return [
            _issue(
                "SEVERE_CONTENT_COLLAPSE",
                "error",
                "polish output dropped too much content",
                {"ratio": round(ratio, 4), "input_chars": source_chars, "output_chars": output_chars},
            )
        ]
    if ratio > POLISH_BLOAT_HARD_RATIO:
        return [
            _issue(
                "SEVERE_CONTENT_BLOAT",
                "error",
                "polish output added too much content",
                {"ratio": round(ratio, 4), "input_chars": source_chars, "output_chars": output_chars},
            )
        ]
    if ratio < POLISH_COLLAPSE_WARNING_RATIO:
        return [
            _issue(
                "CONTENT_COLLAPSE",
                "warning",
                "polish output substantially shortened the chunk",
                {"ratio": round(ratio, 4), "input_chars": source_chars, "output_chars": output_chars},
            )
        ]
    if ratio > POLISH_BLOAT_WARNING_RATIO:
        return [
            _issue(
                "CONTENT_BLOAT",
                "warning",
                "polish output substantially expanded the chunk",
                {"ratio": round(ratio, 4), "input_chars": source_chars, "output_chars": output_chars},
            )
        ]
    return []


def _compact_char_count(text: str) -> int:
    return len("".join(text.split()))


def _parsed_turn_line(turn: ParsedTurn) -> str:
    return (
        f"[{_format_precise_timestamp(turn.start)} - {_format_precise_timestamp(turn.end)}] {turn.speaker}: {turn.text}"
    )


def _format_precise_timestamp(seconds: float) -> str:
    safe = max(0.0, seconds)
    whole_seconds = int(safe)
    hundredths = int(round((safe - whole_seconds) * 100))
    if hundredths == 100:
        whole_seconds += 1
        hundredths = 0
    return f"{whole_seconds // 3600:02d}:{(whole_seconds % 3600) // 60:02d}:{whole_seconds % 60:02d}.{hundredths:02d}"


def _parse_precise_timestamp(value: str) -> float:
    hours_text, minutes_text, seconds_text = value.split(":", 2)
    return int(hours_text) * 3600 + int(minutes_text) * 60 + float(seconds_text)


def _issue(code: str, severity: str, detail: str, value: Any = None) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "severity": severity, "detail": detail}
    if value is not None:
        issue["value"] = value
    return issue


def _issue_label(issue: dict[str, Any]) -> str:
    return f"{issue.get('severity', 'unknown')}:{issue.get('code', 'UNKNOWN')}"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline != -1:
        stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()
