# Agentic RAG - text chunking.
# Recursive character splitter with CJK-safe separators (borrowed from the
# Multilingual-Personal-Diary-AI-System repo's splitter design: the separator
# list deliberately avoids splitting inside words and keeps mixed CJK text
# intact), plus a small overlap so chunk boundaries don't lose context.

from config import settings

# Separators ordered by priority. CJK punctuation is included so Chinese /
# Japanese / Korean text breaks on sentence punctuation, not on spaces.
DEFAULT_SEPARATORS = [
    "\n\n", "\n", "。", "！", "？", "；", "：", "…", "．",
    ".", "!", "?", ";", ":", "，", "、", "·", "—", "–", " ", "",
]


def recursive_split(text, chunk_size=None, chunk_overlap=None, separators=None):
    """Split `text` into chunks <= chunk_size using priority separators,
    then prepend `chunk_overlap` chars of the previous chunk to each next one.
    Defaults come from settings (CHUNK_SIZE / CHUNK_OVERLAP)."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    separators = separators if separators is not None else DEFAULT_SEPARATORS
    final_chunks: list[str] = []

    def _split(text: str, level: int):
        if len(text) <= chunk_size:
            final_chunks.append(text)
            return
        sep = separators[level] if level < len(separators) else ""
        pieces = text.split(sep) if sep else list(text)

        buffer = ""
        for p in pieces:
            if len(p) > chunk_size:
                # A single piece is too long: try deeper separators, else hard cut.
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                if sep and level + 1 < len(separators):
                    _split(p, level + 1)
                else:
                    for i in range(0, len(p), chunk_size):
                        final_chunks.append(p[i:i + chunk_size])
                continue
            if buffer and len(buffer) + len(sep) + len(p) > chunk_size:
                final_chunks.append(buffer)
                buffer = ""
            buffer += (sep if buffer else "") + p
        if buffer:
            final_chunks.append(buffer)

    _split(text, 0)

    # Apply overlap: carry the tail of the previous chunk into the next.
    out = []
    prev = ""
    for c in final_chunks:
        if prev and chunk_overlap > 0:
            c = prev[-chunk_overlap:] + c
        out.append(c)
        prev = c
    return [c for c in out if c.strip()]


def chunk_text(text, chunk_size=None, chunk_overlap=None):
    return recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
