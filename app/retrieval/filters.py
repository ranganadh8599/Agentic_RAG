# Agentic RAG - metadata filtering for retrieval.
#
# Normalizes a caller-supplied filter dict and builds the WHERE / post-filter
# pieces used by every search channel (dense, sparse, keyword, filename).
# Placement follows METADATA_FILTER_MODE: 'pre' (WHERE before the ANN scan)
# or 'post' (fetch more, trim after) — each trades speed vs recall.

from app.core.config import settings


def norm_filters(filters):
    """Normalize a metadata filter dict -> dict or None.

    Supported keys:
      user_id             - documents owned by a user (documents.user_id)
      date_from / date_to - document ingest date range (documents.created_at)
      tags                - list[str]; chunk metadata tags
      tags_mode           - 'any' (default) | 'all'
    """
    if not filters:
        return None
    f = {}
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        # A list of allowed owners; None in the list = the shared/admin-ingested
        # (ownerless) docs a normal user is allowed to see.
        vals = [None if v in (None, "", "None") else str(v) for v in uid]
        if vals:
            f["user_id"] = vals
    elif uid:
        f["user_id"] = str(uid)
    dfrom = filters.get("date_from")
    if dfrom:
        f["date_from"] = str(dfrom)
    dto = filters.get("date_to")
    if dto:
        f["date_to"] = str(dto)
    tags = filters.get("tags")
    if tags:
        f["tags"] = [str(t) for t in tags]
        f["tags_mode"] = "all" if str(filters.get("tags_mode", "any")).lower() == "all" else "any"
    return f or None


def filter_where(filters):
    """Build (where_clause, params) for PRE-filtering the chunks JOIN documents
    query before the ANN scan. user_id/date live on documents, tags on chunk
    metadata (JSONB containment)."""
    where, params = [], []
    if not filters:
        return "", []
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        parts = []
        allowed = [v for v in uid if v is not None]
        if None in uid:
            # Ownerless docs ingested by an admin/CLI = the shared corpus that
            # every normal user may retrieve.
            parts.append("(d.user_id IS NULL AND d.ingested_by IS NOT NULL)")
        if allowed:
            parts.append("d.user_id = ANY(%s)")
            params.append(allowed)
        if parts:
            where.append("(" + " OR ".join(parts) + ")")
    elif uid:
        where.append("d.user_id = %s")
        params.append(uid)
    dfrom = filters.get("date_from")
    if dfrom:
        where.append("d.created_at >= %s")
        params.append(dfrom)
    dto = filters.get("date_to")
    if dto:
        where.append("d.created_at <= %s")
        params.append(dto)
    tags = filters.get("tags")
    if tags:
        op = "?&" if filters.get("tags_mode") == "all" else "?|"
        where.append(f"c.metadata->'tags' {op} %s")
        params.append(tags)
    return (" AND " + " AND ".join(where)) if where else "", params


def is_post_filter(filters) -> bool:
    """True when filters should be applied AFTER retrieval (trim) instead of
    BEFORE the ANN scan. Post-filtering guarantees recall (fetch then trim) but
    scans more; pre-filtering is faster but selective filters can cut ANN
    results (recall risk) — the speed/recall tradeoff the caller chooses."""
    return bool(filters) and settings.METADATA_FILTER_MODE == "post"


def passes_filter(row, filters) -> bool:
    """Python-side filter for POST-filtering. Row must carry user_id, created_at
    and metadata (the search queries add those columns in post mode)."""
    if not filters:
        return True
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        row_uid = row.get("user_id")
        if row_uid is None:
            if None not in uid or not row.get("ingested_by"):
                return False
        elif row_uid not in [v for v in uid if v is not None]:
            return False
    elif uid and row.get("user_id") != uid:
        return False
    dfrom, dto = filters.get("date_from"), filters.get("date_to")
    created = row.get("created_at")
    if (dfrom or dto) and created is not None:
        cdate = str(created)[:10]
        if dfrom and cdate < str(dfrom)[:10]:
            return False
        if dto and cdate > str(dto)[:10]:
            return False
    tags = filters.get("tags")
    if tags:
        row_tags = (row.get("metadata") or {}).get("tags") or []
        if filters.get("tags_mode") == "all":
            if not all(t in row_tags for t in tags):
                return False
        elif not any(t in row_tags for t in tags):
            return False
    return True
