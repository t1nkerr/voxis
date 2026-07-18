from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from voxis.audio import sha256_text
from voxis.config import VoxisConfig
from voxis.llm import call_json_schema_llm, no_thinking_extra_body

MAX_TERMS = 30
MAX_TERM_CHARS = 80
TERM_EXTRACTION_PROMPT_TEMPLATE_ID = "asr_term_extraction_json_v1"
ASR_PROMPT_POLICY_ID = "asr_terms_bare_wordlist_v2"
PROMPT_ECHO_POLICY_ID = "asr_prompt_echo_detector_v1"
MIN_ECHO_TERMS = 4
MIN_ECHO_ORDERED_TERMS = 4
MIN_ECHO_OVERLAP_RATIO = 0.45
TERM_EXTRACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["terms"],
    "additionalProperties": False,
}
_WHITESPACE_RE = re.compile(r"\s+")
_BOUNDARY_PUNCT = " \t\n\r\"'“”‘’「」『』.,;:!?()[]{}<>|"
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_GENERIC_TERMS = {
    "audio",
    "episode",
    "podcast",
    "show notes",
    "transcript",
}

TERM_EXTRACTION_PROMPT = """Extract ASR spelling-hint terms from this context.

Return JSON in this shape:
{{"terms":["term"]}}

Rules:
- Include only visible names or terms that may be spoken aloud and are useful for ASR spelling.
- Keep people, companies, products, models, places, podcasts, organizations, acronyms, technical terms, mixed Chinese-English terms, and Chinese proper names.
- Preserve exact spelling, casing, punctuation, and language from the context.
- Prefer specific multi-word terms over generic single words.
- Do not include generic keywords, section labels, timestamps, URLs, full sentences, contact/footer text, dates, money amounts, episode numbers, or standalone role/title words.
- Do not translate, rewrite, explain, or add terms not present in the context.
- Return at most 30 terms.

<context>
{context}
</context>
"""


@dataclass(frozen=True)
class TermExtractionResult:
    terms: tuple[str, ...]
    status: str
    model: str | None
    prompt_template_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptEchoSegment:
    index: int
    start: float | None
    end: float | None
    term_count: int
    ordered_term_count: int
    term_overlap_ratio: float
    text_excerpt: str


@dataclass(frozen=True)
class PromptEchoResult:
    status: str
    policy_id: str = PROMPT_ECHO_POLICY_ID
    segment_count: int = 0
    suspect_segments: tuple[PromptEchoSegment, ...] = ()


def extract_terms(context: str | None, *, config: VoxisConfig) -> TermExtractionResult:
    if not context:
        return TermExtractionResult(terms=(), status="skipped_no_context", model=None)
    if config.llm_api_key is None:
        return TermExtractionResult(
            terms=(),
            status="skipped_no_llm_key",
            model=config.asr_terms_model,
        )
    prompt = TERM_EXTRACTION_PROMPT.format(context=context.strip())
    try:
        response = call_json_schema_llm(
            prompt,
            config=config,
            model=config.asr_terms_model,
            schema_name="asr_terms_response",
            schema=TERM_EXTRACTION_SCHEMA,
            max_tokens=2048,
            extra_body=no_thinking_extra_body(config),
        )
    except Exception as exc:
        return TermExtractionResult(
            terms=(),
            status="failed_llm_error",
            model=config.asr_terms_model,
            prompt_template_id=TERM_EXTRACTION_PROMPT_TEMPLATE_ID,
            issues=(f"{type(exc).__name__}: {exc}",),
        )
    terms, issues = _parse_llm_terms(response.text)
    if not terms:
        return TermExtractionResult(
            terms=(),
            status="failed_empty_llm_terms",
            model=config.asr_terms_model,
            prompt_template_id=TERM_EXTRACTION_PROMPT_TEMPLATE_ID,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            elapsed_seconds=response.elapsed_seconds,
            issues=tuple(issues),
        )
    return TermExtractionResult(
        terms=terms,
        status="succeeded" if not issues else "succeeded_with_issues",
        model=config.asr_terms_model,
        prompt_template_id=TERM_EXTRACTION_PROMPT_TEMPLATE_ID,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        elapsed_seconds=response.elapsed_seconds,
        issues=tuple(issues),
    )


def term_extraction_payload(result: TermExtractionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "model": result.model,
        "prompt_template_id": result.prompt_template_id,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "terms": list(result.terms),
        "issues": list(result.issues),
    }


def global_terms(config: VoxisConfig) -> tuple[str, ...]:
    return _normalize_prompt_list(config.asr_terms)


def prompt_terms(terms: Iterable[str] | None, *, config: VoxisConfig, max_terms: int = MAX_TERMS) -> tuple[str, ...]:
    return _merge_prompt_terms(global_terms(config), normalize_terms(terms))[:max_terms]


def system_prompt_from_terms(terms: Iterable[str] | None, *, config: VoxisConfig) -> str | None:
    normalized = prompt_terms(terms, config=config)
    if not normalized:
        return None
    return "\n".join(normalized)


def asr_prompt_payload(
    terms: Sequence[str],
    *,
    extracted_terms: Sequence[str],
    global_terms: Sequence[str],
    trimmed_segment_count: int,
) -> dict[str, Any]:
    encoded_terms = json.dumps(list(terms), ensure_ascii=False, separators=(",", ":"))
    return {
        "policy_id": ASR_PROMPT_POLICY_ID,
        "mode": "bare_word_list",
        "max_terms": MAX_TERMS,
        "term_count": len(terms),
        "terms": list(terms),
        "global_terms": list(global_terms),
        "extracted_terms": list(extracted_terms),
        "terms_hash": sha256_text(encoded_terms),
        "trimmed_segment_count": trimmed_segment_count,
    }


def detect_prompt_echo(segments: Sequence[dict[str, Any]], terms: Iterable[str] | None) -> PromptEchoResult:
    normalized_terms = _normalize_prompt_list(terms)
    if not normalized_terms:
        return PromptEchoResult(status="skipped_no_terms", segment_count=len(segments))
    suspect_segments: list[PromptEchoSegment] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text", ""))
        analysis = _prompt_echo_analysis(text, normalized_terms)
        if _is_prompt_echo(analysis):
            suspect_segments.append(
                PromptEchoSegment(
                    index=index,
                    start=_float_or_none(segment.get("start")),
                    end=_float_or_none(segment.get("end")),
                    term_count=analysis["term_count"],
                    ordered_term_count=analysis["ordered_term_count"],
                    term_overlap_ratio=analysis["term_overlap_ratio"],
                    text_excerpt=text[:240],
                )
            )
    return PromptEchoResult(
        status="detected" if suspect_segments else "passed",
        segment_count=len(segments),
        suspect_segments=tuple(suspect_segments),
    )


def prompt_echo_payload(result: PromptEchoResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "policy_id": result.policy_id,
        "segment_count": result.segment_count,
        "suspect_segments": [
            {
                "index": segment.index,
                "start": segment.start,
                "end": segment.end,
                "term_count": segment.term_count,
                "ordered_term_count": segment.ordered_term_count,
                "term_overlap_ratio": round(segment.term_overlap_ratio, 3),
                "text_excerpt": segment.text_excerpt,
            }
            for segment in result.suspect_segments
        ],
    }


def prompt_echo_tail_start(result: PromptEchoResult) -> int | None:
    """Return the first segment index to trim when all suspect segments are a suffix."""
    if not result.suspect_segments:
        return None
    suspect_indexes = {segment.index for segment in result.suspect_segments}
    tail_start = result.segment_count
    while tail_start > 0 and tail_start - 1 in suspect_indexes:
        tail_start -= 1
    if tail_start == result.segment_count:
        return None
    if any(index < tail_start for index in suspect_indexes):
        return None
    return tail_start


def _parse_llm_terms(text: str) -> tuple[tuple[str, ...], list[str]]:
    issues: list[str] = []
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        return (), [f"invalid_json: {exc}"]
    if not isinstance(payload, dict):
        return (), ["bad_shape: expected object"]
    raw_terms = payload.get("terms")
    if not isinstance(raw_terms, list):
        return (), ["bad_shape: expected terms list"]
    string_terms: list[str] = []
    for value in raw_terms:
        if isinstance(value, str):
            string_terms.append(value)
        else:
            issues.append(f"dropped_non_string: {value!r}")
    return normalize_terms(string_terms)[:MAX_TERMS], issues


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


def normalize_terms(terms: Iterable[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in terms or ():
        term = _clean_term(raw)
        if not _term_allowed(term):
            continue
        key = term.casefold()
        if key in seen:
            continue
        if any(_contains(existing, term) for existing in normalized):
            continue
        kept: list[str] = []
        for existing in normalized:
            if _contains(term, existing):
                seen.discard(existing.casefold())
                continue
            kept.append(existing)
        normalized = kept
        normalized.append(term)
        seen.add(key)
    return tuple(normalized)


def _merge_prompt_terms(*term_groups: Iterable[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for terms in term_groups:
        for term in _normalize_prompt_list(terms):
            key = term.casefold()
            if key in seen:
                continue
            merged.append(term)
            seen.add(key)
    return tuple(merged)


def _normalize_prompt_list(terms: Iterable[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in terms or ():
        term = _clean_term(raw)
        if not _term_allowed(term):
            continue
        key = term.casefold()
        if key in seen:
            continue
        normalized.append(term)
        seen.add(key)
    return tuple(normalized)


def _prompt_echo_analysis(text: str, terms: Sequence[str]) -> dict[str, Any]:
    hits: list[str] = []
    text_compact = _compact_text(text)
    ordered_term_count = 0
    ordered_position = -1
    for term in terms:
        if _term_in_text(term, text):
            hits.append(term)
        term_compact = _compact_text(term)
        prompt_position = text_compact.find(term_compact, ordered_position + 1)
        if prompt_position != -1:
            ordered_term_count += 1
            ordered_position = prompt_position
    overlap_chars = sum(len(_compact_text(term)) for term in hits)
    return {
        "term_count": len(hits),
        "ordered_term_count": ordered_term_count,
        "term_overlap_ratio": overlap_chars / max(1, len(text_compact)),
    }


def _is_prompt_echo(analysis: dict[str, Any]) -> bool:
    return (
        int(analysis["term_count"]) >= MIN_ECHO_TERMS
        and int(analysis["ordered_term_count"]) >= MIN_ECHO_ORDERED_TERMS
        and float(analysis["term_overlap_ratio"]) >= MIN_ECHO_OVERLAP_RATIO
    )


def _term_in_text(term: str, text: str) -> bool:
    if _CJK_RE.search(term):
        return _compact_text(term) in _compact_text(text)
    if len(term.split()) > 1 or any(char in "-&." for char in term):
        return _compact_text(term) in _compact_text(text)
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE))


def _clean_term(value: str) -> str:
    term = _WHITESPACE_RE.sub(" ", str(value)).strip(_BOUNDARY_PUNCT)
    if term.endswith("'s"):
        term = term[:-2]
    return term.strip(_BOUNDARY_PUNCT)


def _term_allowed(term: str) -> bool:
    if not (2 <= len(term) <= MAX_TERM_CHARS):
        return False
    if term.casefold() in _GENERIC_TERMS:
        return False
    if term.isdigit() or re.search(r"https?://|www\.|\.com", term, flags=re.IGNORECASE):
        return False
    has_cjk = bool(_CJK_RE.search(term))
    has_signal = has_cjk or any(char.isupper() or char.isdigit() or char in "-&" for char in term)
    return has_signal


def _contains(container: str, term: str) -> bool:
    if container.casefold() == term.casefold():
        return True
    if len(term.split()) > 1:
        return False
    if _CJK_RE.search(container) or _CJK_RE.search(term):
        return term in container and term != container
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    return bool(re.search(pattern, container, flags=re.IGNORECASE))


def _compact_text(text: str) -> str:
    return _WHITESPACE_RE.sub("", text).casefold()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
