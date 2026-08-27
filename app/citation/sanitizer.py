# Agentic RAG - citation sanitization.
#
# Deterministic post-processing that keeps citations honest:
#   * out-of-range [n] (no matching context block) are dropped entirely;
#   * "padding" citations (a block that has no lexical overlap with the claim
#     it is attached to) are pruned from multi-citation groups, always keeping
#     the best-scoring one so a claim retains at least one citation;
#   * duplicate numbers are collapsed and ranges are expanded.
# This is a hard backstop on top of the (soft) Writer/Critic prompt rules.

import re

from app.core.config import settings

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
# Common English stopwords that would otherwise make any two chunks look
# "overlapping" and defeat the padding-citation check.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "there", "their", "they",
    "them", "we", "our", "you", "your", "he", "she", "his", "her", "i", "me",
    "my", "do", "does", "did", "have", "has", "had", "not", "no", "so", "if",
    "then", "than", "about", "into", "over", "under", "also", "can", "could",
    "should", "will", "would", "may", "might", "must", "etc", "use", "used",
    "using", "one", "two", "three", "via", "such", "which", "what", "when",
    "where", "how", "who", "whom", "why", "any", "all", "each", "every", "some",
    "more", "most", "only", "other", "very", "just", "because", "after",
    "before", "while", "during", "per",
}


def significant_words(text):
    """Lowercased significant (non-stopword) words in `text`."""
    return [w.lower() for w in _WORD_RE.findall(text or "")
            if w.lower() not in _STOPWORDS]


def _containing_sentence(text, pos):
    """Return the sentence (roughly) that contains position `pos` in `text`."""
    before = text[:pos]
    start = max(before.rfind(". "), before.rfind("! "), before.rfind("? "),
                before.rfind("\n")) + 1
    after = text[pos:]
    m = re.search(r"[.!?\n]", after)
    end = pos + (m.start() if m else len(after))
    return text[start:end]


def _overlap_fraction(claim_words, chunk_text):
    """Fraction of the claim's significant words that appear in the chunk."""
    if not claim_words:
        return 1.0
    chunk_words = set(significant_words(chunk_text))
    hits = [w for w in claim_words if w in chunk_words]
    return len(hits) / len(claim_words)


def sanitize_citations(answer, blocks):
    """Clean the citation markers in a generated answer (see module docstring)."""
    if not answer:
        return answer
    if not blocks:
        return _CITE_RE.sub("", answer).strip()
    by_num = {r.get("citation"): (r.get("content") or "") for r in blocks}
    parts, last, dropped = [], 0, False
    for m in _CITE_RE.finditer(answer):
        parts.append(answer[last:m.start()])
        nums = []
        for p in re.split(r"[,]", m.group(1)):
            p = p.strip()
            rm = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", p)
            if rm:
                nums.extend(range(int(rm.group(1)), int(rm.group(2)) + 1))
            elif p.isdigit():
                nums.append(int(p))
        nums = list(dict.fromkeys(nums))  # dedupe, keep order
        valid = [n for n in nums if n in by_num]
        if not valid:
            last = m.end()  # drop out-of-range citation entirely
            dropped = True
            continue
        if len(valid) == 1:
            parts.append(f"[{valid[0]}]")
        else:
            claim = significant_words(_containing_sentence(answer, m.start()))
            if not claim:
                parts.append("[" + ",".join(map(str, valid)) + "]")
            else:
                scored = [(_overlap_fraction(claim, by_num[n]), n) for n in valid]
                best_n = max(scored, key=lambda x: (x[0], -x[1]))[1]  # top score; ties -> smallest n
                keep = sorted({n for score, n in scored
                               if score >= settings.CITATION_OVERLAP_THRESHOLD} or {best_n})
                parts.append("[" + ",".join(map(str, keep)) + "]")
        last = m.end()
    parts.append(answer[last:])
    out = "".join(parts).strip()
    if dropped:
        out = re.sub(r"[ \t]{2,}", " ", out)  # tidy space left by a dropped marker
    return out
