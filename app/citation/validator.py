# Agentic RAG - citation validation.
#
# Determines which of the retrieved blocks an answer actually cites and
# verifies those citations are grounded:
#   * the answer's [n] markers are sanitized (dropping out-of-range / padding
#     citations — see sanitizer.py);
#   * only sources actually cited as [n] survive;
#   * stored page images are attached for display.
# This is the deterministic backstop on top of the (soft) Writer/Critic
# prompt rules.

import re

_CITE_SPLIT_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer: str) -> set[int]:
    """All citation numbers [n] actually present in the answer text."""
    cited = set()
    for m in _CITE_SPLIT_RE.finditer(answer or ""):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                cited.add(int(part))
    return cited


def validated_citations(answer, blocks):
    """Sanitize the answer's citations, then keep only sources actually cited
    as [n]. Returns (sources, sanitized_answer) so the caller can also persist
    the cleaned answer (e.g. into the semantic cache). Also attaches the
    page's stored image (if any) so the UI can display it alongside the text,
    even when the cited chunk is a text chunk."""
    from app.citation.formatter import attach_page_images, format_sources
    from app.citation.sanitizer import sanitize_citations

    answer = sanitize_citations(answer or "", blocks)
    cited = cited_numbers(answer)
    sources = [s for s in format_sources(blocks, answer) if s["citation"] in cited]
    return attach_page_images(sources), answer
