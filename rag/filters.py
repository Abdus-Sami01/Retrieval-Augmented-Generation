def passes_filters(source_path: str, tags: list[str], filters: dict | None) -> bool:
    """Shared filter semantics for both dense (FAISS) and keyword (BM25) retrieval,
    so a hybrid search sees identical filtering behavior from either side.
    Recognized keys: 'source_path' (substring match), 'tags' (any-of match)."""
    if not filters:
        return True
    wanted_source = filters.get("source_path")
    if wanted_source and wanted_source not in source_path:
        return False
    wanted_tags = filters.get("tags")
    if wanted_tags and not (set(wanted_tags) & set(tags)):
        return False
    return True
