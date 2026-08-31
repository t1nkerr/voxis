from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from voxis.audio import sha256_text
from voxis.config import VoxisConfig
from voxis.diarization_types import Turn
from voxis.llm import call_json_schema_llm, no_thinking_extra_body

SPEAKER_MAP_SCHEMA_VERSION = "speaker_map_v1"
SPEAKER_MAP_PROMPT_TEMPLATE_ID = "speaker_map_first10_first_appearance_window_v3"
SPEAKER_MAP_STRUCTURAL_ATTEMPTS = 2
SPEAKER_MAP_MIN_RESPONSE_TOKENS = 256
SPEAKER_MAP_MAX_RESPONSE_TOKENS = 2048
SPEAKER_MAP_TOKENS_PER_SPEAKER = 64
SPEAKER_MAP_RESPONSE_EXCERPT_CHARS = 500
PROBE_INITIAL_TURNS = 10
PROBE_FIRST_APPEARANCE_CONTEXT = 2
ROLE_INTRODUCTION_POLICY_ID = "speaker_map_role_introduction_guard_v1"
ROLE_INTRODUCTION_FAILURE_KIND = "role_introduction_conflict"
ROLE_INTRODUCTION_MAX_GAP_SECONDS = 2.5
ROLE_INTRODUCTION_MIN_ACK_WORDS = 3
ROLE_INTRODUCTION_MAX_ACK_PREFIX_CHARS = 32
SPEAKER_LABEL_RE = re.compile(r"SPEAKER_\d+")
UNKNOWN_SPEAKER_RE = re.compile(r"^Unknown \(SPEAKER_\d+\)$")
GENERIC_UNKNOWN_PLACEHOLDERS = {"unknown", "unknown speaker", "n/a", "na", "none", "null"}
ROLE_INTRODUCTION_WELCOME_RE = re.compile(
    r"(?P<phrase>welcome(?:\s+back|\s+to|\b)|thanks?\s+for\s+joining|great\s+to\s+have\s+you)",
    re.IGNORECASE,
)
ROLE_INTRODUCTION_ADDRESS_SUFFIX_RE = re.compile(r"(?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\s*,?\s*$")
ROLE_INTRODUCTION_ACK_RE = re.compile(
    r"(?P<phrase>thank\s+you\s+for\s+(?:having|inviting)\s+me|"
    r"thanks\s+for\s+(?:having|inviting)\s+me|"
    r"(?:happy|glad|excited|pleased|great)\s+to\s+be\s+here|"
    r"pleasure\s+to\s+be\s+here)",
    re.IGNORECASE,
)
ROLE_INTRODUCTION_QUOTED_OR_CLIP_CUE_RE = re.compile(
    r"\b(?:ad|advertisement|clip|quote|quoted|said|says|trailer)\b",
    re.IGNORECASE,
)
CONTEXT_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

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
    role_introduction_evidence: list[dict[str, Any]]
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
    probe = _speaker_map_probe(turns)
    probe_turns = [entry.turn for entry in probe]
    probe_speakers = _speaker_labels_from_turns(probe_turns)
    all_turn_speakers = _speaker_labels_from_turns(turns)
    stripped_context = (context or "").strip()
    role_introduction_evidence = detect_role_introductions(turns, context=stripped_context)
    if not all_turn_speakers:
        return _skipped(
            "skipped_no_speakers",
            probe_speakers=probe_speakers,
            all_turn_speakers=all_turn_speakers,
            role_introduction_evidence=role_introduction_evidence,
        )
    evidence_mode = "context_and_probe" if stripped_context else "probe_only"
    if config.llm_api_key is None:
        return _skipped(
            "skipped_no_llm_key",
            probe_speakers=probe_speakers,
            all_turn_speakers=all_turn_speakers,
            evidence_mode=evidence_mode,
            role_introduction_evidence=role_introduction_evidence,
        )

    prompt = SPEAKER_MAP_PROMPT.format(context=stripped_context, probe=_turns_to_text(probe_turns))
    schema = _speaker_map_response_schema(probe_speakers)
    max_tokens = _speaker_map_max_tokens(probe_speakers)
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
                extra_body=no_thinking_extra_body(config),
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
                role_introduction_evidence=role_introduction_evidence,
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
    base_status = _status_from_speaker_map(raw_mapping=raw_mapping, issues=issues)
    role_introduction_issues = (
        _role_introduction_conflicts(mapping, role_introduction_evidence) if base_status != "failed" else []
    )
    issues.extend(role_introduction_issues)
    status = _status_from_speaker_map(raw_mapping=raw_mapping, issues=issues)
    failure_kind = None
    if role_introduction_issues:
        failure_kind = ROLE_INTRODUCTION_FAILURE_KIND
    elif status == "failed":
        failure_kind = "invalid_output"
    return SpeakerMapResult(
        status=status,
        failure_kind=failure_kind,
        mapping=mapping,
        probe_speakers=probe_speakers,
        all_turn_speakers=all_turn_speakers,
        issues=issues,
        probe_turns=_probe_turns_payload(probe),
        role_introduction_evidence=role_introduction_evidence,
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
        "role_introduction": {
            "policy_id": ROLE_INTRODUCTION_POLICY_ID,
            "evidence": result.role_introduction_evidence,
            "conflict_count": sum(issue.get("code") == "ROLE_INTRODUCTION_CONFLICT" for issue in result.issues),
        },
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
        "role_introduction_evidence_count": len(result.role_introduction_evidence),
        "role_introduction_conflict_count": sum(
            issue.get("code") == "ROLE_INTRODUCTION_CONFLICT" for issue in result.issues
        ),
    }


def _skipped(
    status: str,
    *,
    probe_speakers: list[str],
    all_turn_speakers: list[str],
    evidence_mode: str | None = None,
    role_introduction_evidence: list[dict[str, Any]] | None = None,
) -> SpeakerMapResult:
    return SpeakerMapResult(
        status=status,
        failure_kind=None,
        mapping={},
        probe_speakers=probe_speakers,
        all_turn_speakers=all_turn_speakers,
        issues=[],
        probe_turns=[],
        role_introduction_evidence=list(role_introduction_evidence or []),
        evidence_mode=evidence_mode,
    )


def _thinking_enabled_receipt(config: VoxisConfig) -> bool | None:
    return False if config.llm_disable_thinking else None


def detect_role_introductions(turns: list[Turn], *, context: str | None) -> list[dict[str, Any]]:
    stripped_context = (context or "").strip()
    if not stripped_context:
        return []
    evidence: list[dict[str, Any]] = []
    for index, welcome_turn in enumerate(turns[:-1]):
        for welcome_match in ROLE_INTRODUCTION_WELCOME_RE.finditer(welcome_turn.text):
            addressed_name = _address_before_welcome(
                welcome_turn.text,
                welcome_match,
                context=stripped_context,
            )
            if addressed_name is None:
                continue
            ack_turn = turns[index + 1]
            if ack_turn.speaker == welcome_turn.speaker:
                continue
            gap_seconds = round(ack_turn.start - welcome_turn.end, 3)
            if gap_seconds < -0.25 or gap_seconds > ROLE_INTRODUCTION_MAX_GAP_SECONDS:
                continue
            if len(ack_turn.text.split()) < ROLE_INTRODUCTION_MIN_ACK_WORDS:
                continue
            ack_match = ROLE_INTRODUCTION_ACK_RE.search(ack_turn.text)
            if ack_match is None:
                continue
            ack_prefix = ack_turn.text[: ack_match.start()]
            if len(ack_prefix) > ROLE_INTRODUCTION_MAX_ACK_PREFIX_CHARS:
                continue
            if ROLE_INTRODUCTION_QUOTED_OR_CLIP_CUE_RE.search(ack_prefix):
                continue
            evidence.append(
                {
                    "welcome_turn_index": index,
                    "ack_turn_index": index + 1,
                    "addressed_name": addressed_name,
                    "welcoming_cluster": welcome_turn.speaker,
                    "target_cluster": ack_turn.speaker,
                    "welcome_phrase": welcome_match.group("phrase"),
                    "ack_phrase": ack_match.group("phrase"),
                    "gap_seconds": gap_seconds,
                    "welcome_excerpt": welcome_turn.text[:400],
                    "ack_excerpt": ack_turn.text[:400],
                }
            )
    return evidence


def _address_before_welcome(
    text: str,
    match: re.Match[str],
    *,
    context: str,
) -> str | None:
    prefix = text[: match.start()]
    suffix_match = ROLE_INTRODUCTION_ADDRESS_SUFFIX_RE.search(prefix)
    if suffix_match is None:
        return None
    cue_window = prefix[max(0, suffix_match.start() - 80) : suffix_match.start()]
    if ROLE_INTRODUCTION_QUOTED_OR_CLIP_CUE_RE.search(cue_window):
        return None
    tokens = suffix_match.group("name").split()
    for start in range(len(tokens)):
        candidate = " ".join(tokens[start:])
        resolved = _resolve_context_name(context, candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_context_name(context: str, address: str) -> str | None:
    sanitized_context = CONTEXT_URL_RE.sub(" ", context)
    if not _context_contains_name(sanitized_context, address):
        return None
    if " " in address:
        return address
    candidates = {
        match.group(0).rstrip(".,;:!?")
        for match in re.finditer(
            rf"(?<!\w){re.escape(address)}\s+[A-Z][A-Za-z.'-]+(?!\w)",
            sanitized_context,
        )
    }
    exact_case = {candidate for candidate in candidates if candidate.split()[0].lower() == address.lower()}
    if len(exact_case) == 1:
        return next(iter(exact_case))
    if exact_case:
        return None
    return None


def _context_contains_name(context: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", context, re.IGNORECASE) is not None


def _role_introduction_conflicts(
    mapping: dict[str, str],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for receipt in evidence:
        addressed_name = str(receipt["addressed_name"])
        target_cluster = str(receipt["target_cluster"])
        welcoming_cluster = str(receipt["welcoming_cluster"])
        target_mapping = mapping.get(target_cluster)
        welcoming_mapping = mapping.get(welcoming_cluster)
        reasons = []
        if not _same_real_name(target_mapping, addressed_name):
            reasons.append("target_cluster_name_mismatch")
        if _same_real_name(welcoming_mapping, addressed_name):
            reasons.append("welcoming_cluster_claims_guest")
        if not reasons:
            continue
        issues.append(
            {
                "code": "ROLE_INTRODUCTION_CONFLICT",
                "severity": "error",
                "detail": "speaker map contradicts explicit welcome and acknowledgment evidence",
                "speaker": target_cluster,
                "value": {
                    "reasons": reasons,
                    "addressed_name": addressed_name,
                    "target_cluster": target_cluster,
                    "target_mapping": target_mapping,
                    "welcoming_cluster": welcoming_cluster,
                    "welcoming_mapping": welcoming_mapping,
                    "evidence": receipt,
                },
            }
        )
    return issues


def _same_real_name(value: str | None, expected: str) -> bool:
    if value is None or UNKNOWN_SPEAKER_RE.fullmatch(value):
        return False
    return value.casefold() == expected.casefold()


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
                    "UNSUPPORTED_NAME",
                    "speaker map value must appear in context or transcript probe evidence; falling back to unknown",
                    speaker,
                    value,
                    severity="warning",
                )
            )
            mapping[speaker] = unknown
            continue
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
