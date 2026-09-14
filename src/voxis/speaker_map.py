from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from voxis.audio import sha256_text
from voxis.config import VoxisConfig
from voxis.diarization_types import Turn
from voxis.llm import call_json_schema_llm, no_thinking_extra_body
from voxis.speaker_context import clean_speaker_context

SPEAKER_MAP_SCHEMA_VERSION = "speaker_map_v1"
SPEAKER_MAP_PROMPT_TEMPLATE_ID = "speaker_map_first10_first_appearance_window_v3"
SPEAKER_MAP_STRUCTURAL_ATTEMPTS = 2
SPEAKER_MAP_MIN_RESPONSE_TOKENS = 256
SPEAKER_MAP_MAX_RESPONSE_TOKENS = 2048
SPEAKER_MAP_TOKENS_PER_SPEAKER = 64
SPEAKER_MAP_RESPONSE_EXCERPT_CHARS = 500
PROBE_INITIAL_TURNS = 10
PROBE_FIRST_APPEARANCE_CONTEXT = 2
SPEAKER_LABEL_RE = re.compile(r"SPEAKER_\d+")
UNKNOWN_SPEAKER_RE = re.compile(r"^Unknown \(SPEAKER_\d+\)$")
GENERIC_UNKNOWN_PLACEHOLDERS = {"unknown", "unknown speaker", "n/a", "na", "none", "null"}

SPEAKER_MAP_PROMPT = """Map diarization cluster IDs to visible speaker names.

SPEAKER_XX labels are acoustic clusters; a cluster may mix hosts, ads, clips, or multiple voices. Use <context> and <transcript_probe> as evidence, but do not force names. External context may be empty; in that case, use only <transcript_probe> as identity evidence.

Rules:
- Include every SPEAKER_XX label from <transcript_probe>.
- Default each label to null.
- Use a real name only when the cluster is mostly one identifiable speaker and that cluster's own turns support the identity.
- Use <context> to confirm spelling or role, not to name a cluster by itself.
- Use only names appearing in <context> or <transcript_probe>; preserve exact spelling.
- Use null for weak, mixed, third-person-only, quoted/interview-clip, or host-discussing-someone evidence.
- If a cluster contains intro/outro, ads, clips, multiple voices or multiple self-introductions, use null unless one speaker clearly dominates that cluster.
- Do not assign a context host name to an intro/ad/clip cluster unless that cluster's own words identify that host.
- Diarization may split one real person into multiple SPEAKER_XX clusters.
- Use the same real name for multiple clusters only when each cluster's own turns independently support that same identity or role.
- Do not copy a name from context onto a second cluster just because the name is plausible. If the second cluster is weak, map it to null.

Return only JSON in the shape below:
{{"mapping":{{"SPEAKER_00":"Name","SPEAKER_01":null}}}}

<context>
{context}
</context>

<transcript_probe>
{probe}
</transcript_probe>
"""


@dataclass(frozen=True)
class SpeakerMapResult:
    status: str
    failure_kind: str | None
    mapping: dict[str, str]
    probe_speakers: list[str]
    all_turn_speakers: list[str]
    issues: list[dict[str, Any]]
    probe_turns: list[dict[str, Any]]
    model: str | None = None
    prompt_template_id: str | None = None
    prompt_sha256: str | None = None
    evidence_mode: str | None = None
    thinking_enabled: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class _ProbeTurn:
    index: int
    turn: Turn
    reasons: tuple[str, ...]


def build_speaker_map(
    turns: list[Turn],
    *,
    context: str | None,
    config: VoxisConfig,
) -> SpeakerMapResult:
    dedicated_provider = config.speaker_map_base_url is not None
    if dedicated_provider:
        # Never reuse the shared provider key for a different endpoint.
        config = replace(
            config,
            llm_base_url=config.speaker_map_base_url or config.llm_base_url,
            llm_api_key=config.speaker_map_api_key,
            llm_disable_thinking=False,
        )
    probe = _speaker_map_probe(turns)
    probe_turns = [entry.turn for entry in probe]
    probe_speakers = _speaker_labels_from_turns(probe_turns)
    all_turn_speakers = _speaker_labels_from_turns(turns)
    stripped_context = clean_speaker_context(context or "")
    if not all_turn_speakers:
        return _skipped("skipped_no_speakers", probe_speakers=probe_speakers, all_turn_speakers=all_turn_speakers)
    evidence_mode = "context_and_probe" if stripped_context else "probe_only"
    if config.llm_api_key is None:
        return _skipped(
            "skipped_no_llm_key",
            probe_speakers=probe_speakers,
            all_turn_speakers=all_turn_speakers,
            evidence_mode=evidence_mode,
        )

    prompt = SPEAKER_MAP_PROMPT.format(context=stripped_context, probe=_turns_to_text(probe_turns))
    schema = _speaker_map_response_schema(probe_speakers)
    max_tokens = None if dedicated_provider else _speaker_map_max_tokens(probe_speakers)
    extra_body = (
        {"thinking": {"type": "enabled"}, "reasoning_effort": config.speaker_map_reasoning_effort or "low"}
        if dedicated_provider
        else no_thinking_extra_body(config)
    )
    prompt_tokens = 0
    completion_tokens = 0
    elapsed_seconds = 0.0
    raw_mapping: dict[str, Any] | None = None
    base_issues: list[dict[str, Any]] = []
    for attempt in range(SPEAKER_MAP_STRUCTURAL_ATTEMPTS):
        try:
            response = call_json_schema_llm(
                prompt,
                config=config,
                model=config.speaker_map_model,
                schema_name="speaker_map_response",
                schema=schema,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as exc:
            return SpeakerMapResult(
                status="failed",
                failure_kind="llm_error",
                mapping={},
                probe_speakers=probe_speakers,
                all_turn_speakers=all_turn_speakers,
                issues=[
                    {
                        "code": "LLM_ERROR",
                        "severity": "error",
                        "detail": "speaker map LLM call failed",
                        "value": f"{type(exc).__name__}: {exc}",
                    }
                ],
                probe_turns=_probe_turns_payload(probe),
                model=config.speaker_map_model,
                prompt_template_id=SPEAKER_MAP_PROMPT_TEMPLATE_ID,
                prompt_sha256=sha256_text(prompt),
                evidence_mode=evidence_mode,
                thinking_enabled=_thinking_enabled_receipt(config),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed_seconds,
            )
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        elapsed_seconds += response.elapsed_seconds
        raw_mapping, base_issues = _parse_speaker_map_response(response.text)
        if (
            raw_mapping is None
            and _has_structural_response_issue(base_issues)
            and attempt + 1 < SPEAKER_MAP_STRUCTURAL_ATTEMPTS
        ):
            continue
        break

    mapping, issues = _canonicalize_speaker_map(
        raw_mapping,
        probe_speakers=probe_speakers,
        all_turn_speakers=all_turn_speakers,
        evidence_text=f"{stripped_context}\n{_turns_to_text(probe_turns)}",
        base_issues=base_issues,
    )
    status = _status_from_speaker_map(raw_mapping=raw_mapping, issues=issues)
    return SpeakerMapResult(
        status=status,
        failure_kind="invalid_output" if status == "failed" else None,
        mapping=mapping,
        probe_speakers=probe_speakers,
        all_turn_speakers=all_turn_speakers,
        issues=issues,
        probe_turns=_probe_turns_payload(probe),
        model=config.speaker_map_model,
        prompt_template_id=SPEAKER_MAP_PROMPT_TEMPLATE_ID,
        prompt_sha256=sha256_text(prompt),
        evidence_mode=evidence_mode,
        thinking_enabled=_thinking_enabled_receipt(config),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=elapsed_seconds,
    )


def apply_speaker_map_to_turns(turns: list[Turn], mapping: dict[str, str]) -> list[Turn]:
    if not mapping:
        return turns
    return [
        Turn(
            speaker=mapping.get(turn.speaker, turn.speaker),
            start=turn.start,
            end=turn.end,
            text=turn.text,
        )
        for turn in turns
    ]


def speaker_map_payload(result: SpeakerMapResult) -> dict[str, Any]:
    return {
        "schema_version": SPEAKER_MAP_SCHEMA_VERSION,
        "status": result.status,
        "failure_kind": result.failure_kind,
        "mapping": result.mapping,
        "probe_speakers": result.probe_speakers,
        "all_turn_speakers": result.all_turn_speakers,
        "probe_turns": result.probe_turns,
        "model": result.model,
        "prompt_template_id": result.prompt_template_id,
        "prompt_sha256": result.prompt_sha256,
        "evidence_mode": result.evidence_mode,
        "thinking_enabled": result.thinking_enabled,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "issues": result.issues,
    }


def speaker_map_provider_payload(result: SpeakerMapResult | None, *, enabled: bool) -> dict[str, Any]:
    if result is None:
        return {
            "enabled": enabled,
            "status": "skipped_no_turns" if enabled else "disabled",
            "mapping_count": 0,
        }
    return {
        "enabled": enabled,
        "status": result.status,
        "failure_kind": result.failure_kind,
        "model": result.model,
        "prompt_template_id": result.prompt_template_id,
        "evidence_mode": result.evidence_mode,
        "thinking_enabled": result.thinking_enabled,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "mapping_count": len(result.mapping),
        "issue_count": len(result.issues),
    }


def _skipped(
    status: str,
    *,
    probe_speakers: list[str],
    all_turn_speakers: list[str],
    evidence_mode: str | None = None,
) -> SpeakerMapResult:
    return SpeakerMapResult(
        status=status,
        failure_kind=None,
        mapping={},
        probe_speakers=probe_speakers,
        all_turn_speakers=all_turn_speakers,
        issues=[],
        probe_turns=[],
        evidence_mode=evidence_mode,
    )


def _thinking_enabled_receipt(config: VoxisConfig) -> bool | None:
    if config.speaker_map_base_url is not None:
        return True
    return False if config.llm_disable_thinking else None


def _speaker_map_probe(turns: list[Turn]) -> list[_ProbeTurn]:
    selected: dict[int, set[str]] = {}

    def add_turn(index: int, reason: str) -> None:
        selected.setdefault(index, set()).add(reason)

    for index, turn in enumerate(turns[:PROBE_INITIAL_TURNS]):
        add_turn(index, "initial_turn")
    covered_speakers = set(_speaker_labels_from_turns([turns[index] for index in selected]))
    for index, turn in enumerate(turns[PROBE_INITIAL_TURNS:], start=PROBE_INITIAL_TURNS):
        speaker = turn.speaker
        if not SPEAKER_LABEL_RE.fullmatch(speaker):
            continue
        if speaker in covered_speakers:
            continue
        start = max(0, index - PROBE_FIRST_APPEARANCE_CONTEXT)
        end = min(len(turns), index + PROBE_FIRST_APPEARANCE_CONTEXT + 1)
        for window_index in range(start, end):
            add_turn(window_index, f"first_speaker_appearance_window:{speaker}")
        covered_speakers.add(speaker)
    return [
        _ProbeTurn(index=index, turn=turns[index], reasons=tuple(sorted(selected[index]))) for index in sorted(selected)
    ]


def _probe_turns_payload(probe: list[_ProbeTurn]) -> list[dict[str, Any]]:
    return [
        {
            "index": entry.index,
            "speaker": entry.turn.speaker,
            "start": round(entry.turn.start, 3),
            "end": round(entry.turn.end, 3),
            "reasons": list(entry.reasons),
        }
        for entry in probe
    ]


def _speaker_labels_from_turns(turns: list[Turn]) -> list[str]:
    return sorted({turn.speaker for turn in turns if SPEAKER_LABEL_RE.fullmatch(turn.speaker)})


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n".join(
        f"[{_format_probe_ts(turn.start)} - {_format_probe_ts(turn.end)}] {turn.speaker}: {turn.text}" for turn in turns
    )


def _format_probe_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _speaker_map_response_schema(probe_speakers: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "object",
                "properties": {speaker: {"type": ["string", "null"]} for speaker in probe_speakers},
                "required": list(probe_speakers),
                "additionalProperties": False,
            }
        },
        "required": ["mapping"],
        "additionalProperties": False,
    }


def _speaker_map_max_tokens(probe_speakers: list[str]) -> int:
    return min(
        SPEAKER_MAP_MAX_RESPONSE_TOKENS,
        max(SPEAKER_MAP_MIN_RESPONSE_TOKENS, SPEAKER_MAP_TOKENS_PER_SPEAKER * max(1, len(probe_speakers))),
    )


def _parse_speaker_map_response(text: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, [
            {
                "code": "MALFORMED_JSON",
                "severity": "error",
                "detail": "speaker map LLM response was not valid JSON",
                "value": str(exc),
                "response_length": len(cleaned),
                "response_excerpt": _truncate_response(cleaned),
            }
        ]
    if not isinstance(parsed, dict):
        return None, [
            {
                "code": "BAD_SHAPE",
                "severity": "error",
                "detail": "speaker map LLM response was not a JSON object",
                "value": cleaned[:200],
                "response_length": len(cleaned),
                "response_excerpt": _truncate_response(cleaned),
            }
        ]
    mapping = parsed.get("mapping")
    if not isinstance(mapping, dict):
        return None, [
            {
                "code": "BAD_SHAPE",
                "severity": "error",
                "detail": "speaker map JSON did not contain an object mapping",
                "value": cleaned[:200],
                "response_length": len(cleaned),
                "response_excerpt": _truncate_response(cleaned),
            }
        ]
    return mapping, []


def _has_structural_response_issue(issues: list[dict[str, Any]]) -> bool:
    return any(issue.get("code") in {"MALFORMED_JSON", "BAD_SHAPE"} for issue in issues)


def _truncate_response(text: str) -> str:
    if len(text) <= SPEAKER_MAP_RESPONSE_EXCERPT_CHARS:
        return text
    half = SPEAKER_MAP_RESPONSE_EXCERPT_CHARS // 2
    return f"{text[:half]}…{text[-half:]}"


def _canonicalize_speaker_map(
    raw_mapping: dict[str, Any] | None,
    *,
    probe_speakers: list[str],
    all_turn_speakers: list[str],
    evidence_text: str,
    base_issues: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    issues = list(base_issues or [])
    probe_set = set(probe_speakers)
    if raw_mapping is None:
        return {}, issues
    raw = raw_mapping

    for key, value in sorted(raw.items(), key=lambda item: str(item[0])):
        speaker = str(key)
        if not SPEAKER_LABEL_RE.fullmatch(speaker):
            issues.append(
                _issue(
                    "EXTRA_KEY",
                    "speaker map returned a non-speaker key; dropping it",
                    speaker,
                    value,
                    severity="warning",
                )
            )
        elif speaker not in probe_set:
            issues.append(
                _issue(
                    "EXTRA_SPEAKER_KEY",
                    "speaker map returned a speaker key that was not present in the prompt probe; dropping it",
                    speaker,
                    value,
                    severity="warning",
                )
            )

    mapping: dict[str, str] = {}
    for speaker in all_turn_speakers:
        unknown = _unknown_speaker_value(speaker)
        if speaker not in probe_set:
            issues.append(
                _issue(
                    "NOT_IN_PROBE",
                    "speaker appears outside the speaker-map prompt probe",
                    speaker,
                    severity="warning",
                )
            )
            mapping[speaker] = unknown
            continue
        if speaker not in raw:
            issues.append(_issue("MISSING_SPEAKER_KEY", "speaker map omitted a probe speaker", speaker))
            continue
        value = raw[speaker]
        if value is None:
            mapping[speaker] = unknown
            continue
        if not isinstance(value, str) or not value.strip():
            issues.append(_issue("BAD_VALUE", "speaker map value must be a non-empty string or null", speaker, value))
            continue
        normalized = value.strip()
        if normalized.casefold() in GENERIC_UNKNOWN_PLACEHOLDERS:
            issues.append(
                _issue("INVALID_UNKNOWN_VALUE", "unknown speaker-map value must be JSON null", speaker, value)
            )
            continue
        if normalized.startswith("Unknown"):
            issues.append(
                _issue("INVALID_UNKNOWN_VALUE", "unknown speaker-map value must be JSON null", speaker, value)
            )
            continue
        if SPEAKER_LABEL_RE.fullmatch(normalized):
            issues.append(
                _issue("RAW_SPEAKER_LABEL_VALUE", "speaker map value cannot be a raw speaker label", speaker, value)
            )
            continue
        if SPEAKER_LABEL_RE.search(normalized) and normalized != unknown:
            issues.append(
                _issue("INVALID_UNKNOWN_VALUE", "unknown speaker-map value must be JSON null", speaker, value)
            )
            continue
        if normalized != unknown and normalized not in evidence_text:
            issues.append(
                _issue(
                    "NAME_NOT_LITERAL",
                    "name does not appear verbatim in context or transcript probe; preserving model output",
                    speaker,
                    value,
                    severity="warning",
                )
            )
        mapping[speaker] = normalized

    lower_to_speakers: dict[str, list[str]] = {}
    for speaker, value in mapping.items():
        if UNKNOWN_SPEAKER_RE.fullmatch(value):
            continue
        lower_to_speakers.setdefault(value.casefold(), []).append(speaker)
    for speakers in lower_to_speakers.values():
        if len(speakers) > 1:
            for speaker in speakers:
                value = mapping[speaker]
                issues.append(
                    _issue(
                        "DUPLICATE_REAL_NAME",
                        "multiple generated speakers map to the same real name; preserving mapping as a warning",
                        speaker,
                        value,
                        severity="warning",
                    )
                )
    return mapping, issues


def _status_from_speaker_map(*, raw_mapping: dict[str, Any] | None, issues: list[dict[str, Any]]) -> str:
    if raw_mapping is None:
        return "failed"
    if any(issue.get("severity") == "error" for issue in issues):
        return "failed"
    if issues:
        return "succeeded_with_issues"
    return "succeeded"


def _issue(
    code: str,
    detail: str,
    speaker: str,
    value: Any = None,
    *,
    severity: str = "error",
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "detail": detail,
        "speaker": speaker,
    }
    if value is not None:
        issue["value"] = value
    return issue


def _unknown_speaker_value(speaker: str) -> str:
    return f"Unknown ({speaker})"


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
