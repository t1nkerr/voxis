"""Remove link destinations from the mapper's copy of external context."""

from __future__ import annotations

import re
from html.parser import HTMLParser


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")


def clean_speaker_context(context: str) -> str:
    # Resolve Markdown reference labels before deleting destination definitions.
    definitions = re.compile(r"(?m)^ {0,3}\[([^\]\n]+)\]:\s*(?:<?https?://|<?www\.)[^\n]*$", re.I)
    references = {m.group(1).casefold() for m in definitions.finditer(context)}
    text = definitions.sub("", context)
    text = re.sub(
        r"\[([^\]\n]+)\]\[([^\]\n]*)\]",
        lambda m: m.group(1) if (m.group(2) or m.group(1)).casefold() in references else m.group(0),
        text,
    )
    # HTML destinations live in attributes; only visible anchor text is retained.
    text = re.sub(r"<(?:https?://|www\.)[^>]+>", "", text, flags=re.I)
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    text = "".join(parser.parts)

    def remove_url(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = ""
        while url and url[-1] in ".,;:!?，。；！":
            trailing = url[-1] + trailing
            url = url[:-1]
        while url.endswith(")") and url.count(")") > url.count("("):
            trailing = ")" + trailing
            url = url[:-1]
        return trailing

    text = re.sub(r"(?:https?://|www\.)[^\s<>\"']+", remove_url, text, flags=re.I)
    text = re.sub(r"\[([^\]\n]+)\]\(\s*\)", r"\1", text)
    return text.strip()
