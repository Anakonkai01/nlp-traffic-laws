"""
Retrieval v4: structure-aware hybrid legal RAG.

This retriever avoids answer/intent-specific rules. It combines independent,
generic rank lists (dense FAISS, BM25, document aliases, article mentions) with
weighted Reciprocal Rank Fusion, then optionally expands neighbouring chunks in
the same legal article.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from langchain_core.documents import Document

ARTICLE_RE = re.compile(r"[Đđ]i\S*u\s+(\d+)")
DOC_NUM_RE = re.compile(r"\b(\d{1,4})[/-](\d{4})(?:[/: -]*(n[đd]|nd|tt|qh|qcvn|bg[t]?vt|bca|cp))?", re.IGNORECASE)
DOC_ID_NUM_RE = re.compile(r"^(nd|tt|luat|qcvn)_(\d{1,4})_(\d{4})")

QUERY_EXPANSIONS = {
    "gplx": "giấy phép lái xe",
    "nđ": "nghị định",
    "nd": "nghị định",
    "tt": "thông tư",
    "xe máy": "xe mô tô xe gắn máy",
    "mô tô": "xe máy xe gắn máy",
    "ô tô": "xe hơi xe ô tô",
    "xe hơi": "ô tô xe ô tô",
    "mức phạt": "xử phạt phạt tiền",
    "bị phạt": "xử phạt phạt tiền",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class RetrievalV4Config:
    dense_k: int = field(default_factory=lambda: _env_int("RAG_V4_DENSE_K", 50))
    sparse_k: int = field(default_factory=lambda: _env_int("RAG_V4_SPARSE_K", 50))
    fuse_k: int = field(default_factory=lambda: _env_int("RAG_V4_FUSE_K", 40))
    ce_k: int = field(default_factory=lambda: _env_int("RAG_V4_CE_K", 12))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_V4_RRF_K", 60))
    w_dense: float = field(default_factory=lambda: _env_float("RAG_V4_W_DENSE", 0.6))
    w_sparse: float = field(default_factory=lambda: _env_float("RAG_V4_W_SPARSE", 0.4))
    w_alias: float = field(default_factory=lambda: _env_float("RAG_V4_W_ALIAS", 0.03))
    w_article: float = field(default_factory=lambda: _env_float("RAG_V4_W_ARTICLE", 0.07))
    use_query_expansion: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_QUERY_EXPANSION", True))
    use_bm25: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_BM25", True))
    use_alias: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_ALIAS", True))
    use_article: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_ARTICLE", True))
    use_doc_aggregation: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_DOC_AGGREGATION", False))
    doc_aggregation_top_n: int = field(default_factory=lambda: _env_int("RAG_V4_DOC_AGGREGATION_TOP_N", 4))
    use_article_window: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_ARTICLE_WINDOW", False))
    article_window_radius: int = field(default_factory=lambda: _env_int("RAG_V4_ARTICLE_WINDOW_RADIUS", 1))
    article_window_policy: str = field(default_factory=lambda: os.environ.get("RAG_V4_ARTICLE_WINDOW_POLICY", "selective"))
    use_ce: bool = field(default_factory=lambda: _env_bool("RAG_V4_USE_CE", False))
    ce_model: str | None = None


def _doc_id(doc: Document) -> str | None:
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _chunk_id(doc: Document) -> str:
    md = doc.metadata or {}
    return str(md.get("chunk_id") or f"{_doc_id(doc) or 'unknown'}:{hash((doc.page_content or '')[:200]) & 0xffffffff:08x}")


def _vectorstore_docs(vs) -> list[Document]:
    docstore = getattr(vs, "docstore", None)
    return list(getattr(docstore, "_dict", {}).values()) if docstore else []


def _dense_search(vs, query: str, k: int) -> list[Document]:
    try:
        return list(vs.similarity_search(query, k=k))
    except Exception as exc:
        print(f"[retrieval_v4] dense unavailable: {exc}")
        return []


def _get_bm25(vs):
    try:
        from sparse_bm25 import BM25Retriever
        return BM25Retriever.load_or_build(vs)
    except Exception as exc:
        print(f"[retrieval_v4] BM25 unavailable: {exc}")
        return None


def _sparse_search(query: str, k: int, bm25) -> list[Document]:
    if bm25 is None:
        return []
    return [doc for doc, _score in bm25.search(query, k=k)]


def expand_query_generic(question: str) -> str:
    """Generic lexical normalization; never maps a query to a specific article/answer."""
    q = question or ""
    low = q.lower()
    additions = [phrase for trigger, phrase in QUERY_EXPANSIONS.items() if trigger in low and phrase not in low]
    return q if not additions else q + " " + " ".join(additions)


def _aliases_from_doc(doc: Document) -> set[str]:
    md = doc.metadata or {}
    did = str(_doc_id(doc) or "").lower()
    title = str(md.get("title") or md.get("article") or "").lower()
    text = f"{did} {title}"
    aliases: set[str] = set()
    for num, year, suffix in DOC_NUM_RE.findall(text):
        aliases.add(f"{num}/{year}")
        aliases.add(f"{num}-{year}")
        suffix_l = (suffix or "").lower()
        if suffix_l in {"nđ", "nd", "cp"} or did.startswith("nd_") or "nghị định" in title:
            aliases.update({f"nghị định {num}", f"{num}/{year}/nđ-cp", f"{num}/{year}/nd-cp"})
        if suffix_l == "tt" or did.startswith("tt_") or "thông tư" in title:
            aliases.update({f"thông tư {num}", f"{num}/{year}/tt"})
        if suffix_l.startswith("qh") or did.startswith("luat_") or "luật" in title:
            aliases.update({f"luật {num}", f"{num}/{year}/qh15"})
        if did.startswith("qcvn_") or "qcvn" in title or "quy chuẩn" in title:
            aliases.update({f"qcvn {num}", f"qcvn {num}:{year}", f"{num}:{year}"})
    m = DOC_ID_NUM_RE.search(did)
    if m:
        kind, num, year = m.groups()
        aliases.update({f"{num}/{year}", f"{num}-{year}"})
        if kind == "nd":
            aliases.add(f"nghị định {num}")
        elif kind == "tt":
            aliases.add(f"thông tư {num}")
        elif kind == "luat":
            aliases.add(f"luật {num}")
        elif kind == "qcvn":
            aliases.add(f"qcvn {num}")
    if "luật đường bộ" in title:
        aliases.add("luật đường bộ")
    if "trật tự" in title and "giao thông" in title:
        aliases.add("luật trật tự")
    if "quy chuẩn" in title and "báo hiệu" in title:
        aliases.add("quy chuẩn báo hiệu")
    return {a for a in aliases if a}


def _build_alias_map(all_docs: Sequence[Document]) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    for doc in all_docs:
        did = _doc_id(doc)
        if did:
            alias_map.setdefault(did, set()).update(_aliases_from_doc(doc))
    return alias_map


def _alias_rank_list(question: str, all_docs: Sequence[Document], cap: int, alias_map: dict[str, set[str]] | None = None) -> list[Document]:
    q = question.lower()
    alias_map = alias_map or _build_alias_map(all_docs)
    matched = {doc_id for doc_id, aliases in alias_map.items() if any(a in q for a in aliases)}
    if not matched:
        return []
    return [doc for doc in all_docs if _doc_id(doc) in matched][:cap]

def _article_rank_list(question: str, all_docs: Sequence[Document], cap: int) -> list[Document]:
    match = ARTICLE_RE.search(question)
    if not match:
        return []
    article = match.group(1)
    return [doc for doc in all_docs if str((doc.metadata or {}).get("article_number") or "") == article][:cap]


def weighted_rrf(named_rank_lists: list[tuple[str, float, list[Document]]], rrf_k: int, top_m: int) -> list[Document]:
    scored = weighted_rrf_scores(named_rank_lists, rrf_k)
    return _materialize_scored(scored, top_m)


def weighted_rrf_scores(named_rank_lists: list[tuple[str, float, list[Document]]], rrf_k: int):
    scores: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    handles: dict[str, Document] = {}
    for name, weight, rank_list in named_rank_lists:
        if weight <= 0:
            continue
        for rank, doc in enumerate(rank_list, start=1):
            cid = _chunk_id(doc)
            handles.setdefault(cid, doc)
            contribution = weight / (rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            parts.setdefault(cid, {})[name] = contribution
    return {cid: (scores[cid], handles[cid], parts.get(cid, {})) for cid in scores}


def _materialize_scored(scored, top_m: int) -> list[Document]:
    ordered = sorted(scored, key=lambda cid: -scored[cid][0])[:top_m]
    out: list[Document] = []
    for cid in ordered:
        score, doc, part = scored[cid]
        md = dict(doc.metadata or {})
        md["retrieval_v4_score"] = round(score, 8)
        md["retrieval_v4_parts"] = {k: round(v, 8) for k, v in sorted(part.items())}
        out.append(Document(page_content=doc.page_content, metadata=md))
    return out


def _apply_doc_aggregation(scored, top_docs: int):
    doc_scores: dict[str, float] = {}
    for _cid, (score, doc, _parts) in scored.items():
        did = _doc_id(doc) or "unknown"
        doc_scores[did] = doc_scores.get(did, 0.0) + score
    keep = {did for did, _ in sorted(doc_scores.items(), key=lambda item: -item[1])[:top_docs]}
    filtered = {cid: val for cid, val in scored.items() if (_doc_id(val[1]) or "unknown") in keep}
    return filtered or scored


_ce_cache: dict[str, object] = {}


def _load_ce(model_id: str | None):
    name = model_id or os.environ.get("RETRIEVAL_CE_MODEL") or "BAAI/bge-reranker-v2-m3"
    if name not in _ce_cache:
        from sentence_transformers import CrossEncoder
        _ce_cache[name] = CrossEncoder(name, max_length=512)
    return _ce_cache[name]


def _ce_rerank(query: str, docs: list[Document], top_t: int, model_id: str | None) -> list[Document]:
    if not docs:
        return docs
    ce = _load_ce(model_id)
    scores = ce.predict([(query, doc.page_content or "") for doc in docs], batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda item: -float(item[1]))[:top_t]
    out: list[Document] = []
    for doc, score in ranked:
        md = dict(doc.metadata or {})
        md["retrieval_v4_ce_score"] = float(score)
        out.append(Document(page_content=doc.page_content, metadata=md))
    return out


def _expand_article_window(
    seeds: list[Document],
    all_docs: Sequence[Document],
    radius: int,
    question: str = "",
    policy: str = "selective",
) -> list[Document]:
    by_key: dict[tuple[str | None, str], list[Document]] = {}
    for doc in all_docs:
        md = doc.metadata or {}
        key = (md.get("doc_id") or md.get("source"), str(md.get("article_number") or ""))
        by_key.setdefault(key, []).append(doc)
    for docs in by_key.values():
        docs.sort(key=lambda doc: str((doc.metadata or {}).get("chunk_id") or ""))

    out: list[Document] = []
    seen: set[str] = set()

    def add(doc: Document):
        cid = _chunk_id(doc)
        if cid not in seen:
            seen.add(cid)
            out.append(doc)

    query_article = ""
    match = ARTICLE_RE.search(question or "")
    if match:
        query_article = match.group(1)
    article_votes: dict[tuple[str | None, str], int] = {}
    for seed in seeds[:10]:
        md = seed.metadata or {}
        key = (md.get("doc_id") or md.get("source"), str(md.get("article_number") or ""))
        article_votes[key] = article_votes.get(key, 0) + 1

    for seed in seeds:
        add(seed)
        md = seed.metadata or {}
        key = (md.get("doc_id") or md.get("source"), str(md.get("article_number") or ""))
        if not key[1]:
            continue
        should_expand = policy == "all" or key[1] == query_article or article_votes.get(key, 0) >= 2
        if not should_expand:
            continue
        siblings = by_key.get(key, [])
        try:
            idx = next(i for i, doc in enumerate(siblings) if _chunk_id(doc) == _chunk_id(seed))
        except StopIteration:
            continue
        for pos in range(max(0, idx - radius), min(len(siblings), idx + radius + 1)):
            add(siblings[pos])
    return out


def build_retriever(vs, cfg: RetrievalV4Config | None = None) -> Callable[[str, int], list[Document]]:
    cfg = cfg or RetrievalV4Config()
    bm25 = _get_bm25(vs) if cfg.use_bm25 else None
    all_docs = _vectorstore_docs(vs)
    alias_map = _build_alias_map(all_docs) if cfg.use_alias else {}

    def retrieve(question: str, top_k: int) -> list[Document]:
        search_query = expand_query_generic(question) if cfg.use_query_expansion else question
        rank_lists: list[tuple[str, float, list[Document]]] = [
            ("dense", cfg.w_dense, _dense_search(vs, search_query, cfg.dense_k)),
        ]
        if cfg.use_bm25:
            rank_lists.append(("bm25", cfg.w_sparse, _sparse_search(search_query, cfg.sparse_k, bm25)))
        if cfg.use_alias:
            rank_lists.append(("alias", cfg.w_alias, _alias_rank_list(question, all_docs, cfg.dense_k, alias_map)))
        if cfg.use_article:
            rank_lists.append(("article", cfg.w_article, _article_rank_list(question, all_docs, cfg.dense_k)))

        scored = weighted_rrf_scores(rank_lists, cfg.rrf_k)
        if cfg.use_doc_aggregation:
            scored = _apply_doc_aggregation(scored, cfg.doc_aggregation_top_n)
        fused = _materialize_scored(scored, cfg.fuse_k)
        if cfg.use_ce:
            fused = _ce_rerank(question, fused, cfg.ce_k, cfg.ce_model)
        if cfg.use_article_window:
            fused = _expand_article_window(
                fused,
                all_docs,
                cfg.article_window_radius,
                question=question,
                policy=cfg.article_window_policy,
            )

        seen: set[str] = set()
        out: list[Document] = []
        for doc in fused:
            cid = _chunk_id(doc)
            if cid in seen:
                continue
            seen.add(cid)
            out.append(doc)
            if len(out) >= top_k:
                break
        return out

    return retrieve


def retrieve_ranked_docs(vs, question: str, top_k: int = 3, candidate_k: int = 30, min_score: float | None = None) -> list[Document]:
    cfg = RetrievalV4Config(fuse_k=max(candidate_k, _env_int("RAG_V4_FUSE_K", 40)))
    return build_retriever(vs, cfg)(question, top_k=top_k)


# Compatibility stubs for older diagnostics.
def classify_intents(question: str) -> list[str]:  # noqa: ARG001
    return []


def expand_traffic_query(question: str) -> str:
    return expand_query_generic(question)


def is_supported_traffic_question(question: str) -> bool:  # noqa: ARG001
    return True
