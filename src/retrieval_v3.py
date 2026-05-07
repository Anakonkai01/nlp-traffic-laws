"""
v3 retrieval — learned + rank-fusion, no eval-fit rules.

Pipeline (each step optional via RetrievalConfig):
  Dense FAISS (BGE-M3, top-N1)
  + BM25 sparse (top-N2)
  + alias rank list (doc-aliases — generic, not eval-fit)
  + article-number rank list (chunks where article_number == query article)
  → Reciprocal Rank Fusion (k=60, top-M)
  → Cross-encoder rerank (pretrained or fine-tuned, top-T)
  → Article-window expansion + dedup → top-K

Public API (drop-in for v2 ``retrieve_ranked_docs``):
    from retrieval_v3 import retrieve_ranked_docs
    docs = retrieve_ranked_docs(vs, q, top_k=5, candidate_k=30, min_score=None)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from langchain_core.documents import Document

DOC_ID_ALIASES = {
    "nd_168_2024_nd_cp": ["nghị định 168", "168/2024", "168-2024", "168/2024/nđ-cp", "168/2024/nd-cp"],
    "nd_158_2024_nd_cp": ["nghị định 158", "158/2024", "158-2024"],
    "nd_165_2024_nd_cp": ["nghị định 165", "165/2024", "165-2024"],
    "nd_336_2025_nd_cp": ["nghị định 336", "336/2025", "336-2025"],
    "tt_65_2024_tt_bca": ["thông tư 65", "65/2024", "65-2024"],
    "tt_79_2024_tt_bca": ["thông tư 79", "79/2024", "79-2024"],
    "tt_05_2025_tt_bgtvt": ["thông tư 05", "thông tư 12", "05/2025", "12/2025"],
    "luat_35_2024_qh15": ["luật 35", "35/2024", "35-2024", "luật đường bộ"],
    "luat_36_2024_qh15": ["luật 36", "36/2024", "36-2024", "luật trật tự"],
    "qcvn_41_2024_bgtvt": ["qcvn 41", "41:2024", "quy chuẩn 41"],
    "nd_39_2023_nd_cp": ["nghị định 39", "39/2023", "đấu giá biển số"],
    "tt_suckhoe_lai_xe": ["sức khỏe lái xe", "khám sức khỏe", "tiêu chuẩn sức khỏe"],
}

ARTICLE_RE = re.compile(r"[Đđ]i[eê]u\s+(\d+)")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class RetrievalConfig:
    dense_k: int = field(default_factory=lambda: _env_int("RAG_DENSE_K", 50))
    sparse_k: int = field(default_factory=lambda: _env_int("RAG_SPARSE_K", 50))
    fuse_k: int = field(default_factory=lambda: _env_int("RAG_FUSE_K", 30))
    ce_k: int = field(default_factory=lambda: _env_int("RAG_CE_K", 10))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    use_bm25: bool = True
    use_alias: bool = False
    use_ce: bool = False
    ce_model: str | None = None
    use_article_window: bool = False
    use_article_constraint: bool = False
    article_window_radius: int = 1


def _doc_alias_matches(question: str) -> set[str]:
    q = question.lower()
    matched = set()
    for doc_id, aliases in DOC_ID_ALIASES.items():
        if any(a in q for a in aliases):
            matched.add(doc_id)
    return matched


def _doc_id(doc: Document) -> str | None:
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _chunk_id(doc: Document) -> str:
    md = doc.metadata or {}
    cid = md.get("chunk_id")
    if cid:
        return str(cid)
    text = (doc.page_content or "")[:200]
    return f"{_doc_id(doc) or 'unknown'}:{hash(text) & 0xffffffff:08x}"


def _vectorstore_docs(vs) -> list[Document]:
    docstore = getattr(vs, "docstore", None)
    docs = getattr(docstore, "_dict", {}) if docstore else {}
    return list(docs.values())


def _dense_search(vs, query: str, k: int) -> list[Document]:
    try:
        return list(vs.similarity_search(query, k=k))
    except Exception:
        return []


def _sparse_search(query: str, k: int, bm25) -> list[Document]:
    if bm25 is None:
        return []
    return [d for d, _ in bm25.search(query, k=k)]


def _alias_rank_list(question: str, all_docs: Sequence[Document], cap: int) -> list[Document]:
    """Generic doc-aliases: chunks whose doc_id matches an alias mention in q.
    Order preserved as encountered (no per-rule magic numbers)."""
    matched = _doc_alias_matches(question)
    if not matched:
        return []
    return [d for d in all_docs if _doc_id(d) in matched][:cap]


def _article_rank_list(question: str, all_docs: Sequence[Document], cap: int) -> list[Document]:
    """Chunks whose article_number matches the article number mentioned in q."""
    m = ARTICLE_RE.search(question)
    if not m:
        return []
    art = m.group(1)
    return [d for d in all_docs if str((d.metadata or {}).get("article_number") or "") == art][:cap]


def reciprocal_rank_fusion(rank_lists: list[list[Document]], rrf_k: int, top_m: int) -> list[Document]:
    scores: dict[str, float] = {}
    handles: dict[str, Document] = {}
    for ranks in rank_lists:
        for rank, doc in enumerate(ranks, start=1):
            cid = _chunk_id(doc)
            handles.setdefault(cid, doc)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.keys(), key=lambda c: -scores[c])
    return [handles[c] for c in ordered[:top_m]]


# --- cross-encoder ------------------------------------------------------------
_ce_cache: dict[str, object] = {}


def _load_ce(model_id: str | None):
    name = model_id or os.environ.get("RETRIEVAL_CE_MODEL") or "BAAI/bge-reranker-v2-m3"
    if name in _ce_cache:
        return _ce_cache[name]
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(name, max_length=512)
    _ce_cache[name] = ce
    return ce


def _ce_rerank(query: str, docs: list[Document], top_t: int, model_id: str | None) -> list[Document]:
    if not docs:
        return docs
    ce = _load_ce(model_id)
    pairs = [(query, d.page_content or "") for d in docs]
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: -float(x[1]))
    return [d for d, _ in ranked[:top_t]]


# --- article-window expansion -------------------------------------------------
def _expand_article_window(seeds: list[Document], all_docs: Sequence[Document], radius: int) -> list[Document]:
    """For each seed, add neighbour chunks with same (doc_id, article_number) ordered by chunk_id.
    Keeps original order of seeds; appends neighbours after each seed."""
    by_key: dict[tuple, list[Document]] = {}
    for d in all_docs:
        md = d.metadata or {}
        key = (md.get("doc_id") or md.get("source"), str(md.get("article_number") or ""))
        by_key.setdefault(key, []).append(d)
    for v in by_key.values():
        v.sort(key=lambda x: str((x.metadata or {}).get("chunk_id") or ""))

    out: list[Document] = []
    seen: set[str] = set()

    def add(doc: Document):
        cid = _chunk_id(doc)
        if cid in seen:
            return
        seen.add(cid)
        out.append(doc)

    for seed in seeds:
        add(seed)
        md = seed.metadata or {}
        key = (md.get("doc_id") or md.get("source"), str(md.get("article_number") or ""))
        if not key[1]:
            continue
        siblings = by_key.get(key, [])
        try:
            idx = next(i for i, x in enumerate(siblings) if _chunk_id(x) == _chunk_id(seed))
        except StopIteration:
            continue
        for j in range(max(0, idx - radius), min(len(siblings), idx + radius + 1)):
            add(siblings[j])
    return out


# --- public API ---------------------------------------------------------------
_bm25_cache = {"vs_id": None, "bm": None}


def _get_bm25(vs):
    if not vs:
        return None
    if _bm25_cache["vs_id"] is id(vs) and _bm25_cache["bm"] is not None:
        return _bm25_cache["bm"]
    try:
        from sparse_bm25 import BM25Retriever

        bm = BM25Retriever.load_or_build(vs)
        _bm25_cache["vs_id"] = id(vs)
        _bm25_cache["bm"] = bm
        return bm
    except Exception as exc:
        print(f"[retrieval_v3] BM25 unavailable: {exc}")
        return None


def build_retriever(vs, cfg: RetrievalConfig | None = None) -> Callable[[str, int], list[Document]]:
    cfg = cfg or RetrievalConfig()
    bm25 = _get_bm25(vs) if cfg.use_bm25 else None
    all_docs = _vectorstore_docs(vs) if (cfg.use_alias or cfg.use_article_constraint or cfg.use_article_window) else []

    def retrieve(question: str, top_k: int) -> list[Document]:
        rank_lists: list[list[Document]] = []
        rank_lists.append(_dense_search(vs, question, cfg.dense_k))
        if cfg.use_bm25 and bm25 is not None:
            rank_lists.append(_sparse_search(question, cfg.sparse_k, bm25))
        if cfg.use_alias:
            al = _alias_rank_list(question, all_docs, cap=cfg.dense_k)
            if al:
                rank_lists.append(al)
        if cfg.use_article_constraint:
            ar = _article_rank_list(question, all_docs, cap=cfg.dense_k)
            if ar:
                rank_lists.append(ar)

        fused = reciprocal_rank_fusion(rank_lists, cfg.rrf_k, cfg.fuse_k)

        if cfg.use_ce and fused:
            fused = _ce_rerank(question, fused, cfg.ce_k, cfg.ce_model)

        if cfg.use_article_window and fused:
            fused = _expand_article_window(fused, all_docs, cfg.article_window_radius)

        seen: set[str] = set()
        out: list[Document] = []
        for d in fused:
            cid = _chunk_id(d)
            if cid in seen:
                continue
            seen.add(cid)
            out.append(d)
            if len(out) >= top_k:
                break
        return out

    return retrieve


def retrieve_ranked_docs(vs, question: str, top_k: int = 3,
                          candidate_k: int = 30, min_score: float | None = None) -> list[Document]:
    """Drop-in replacement for v2 API. Reads RetrievalConfig from env vars."""
    cfg = RetrievalConfig(
        fuse_k=max(candidate_k, _env_int("RAG_FUSE_K", 30)),
        use_bm25=os.environ.get("RAG_USE_BM25", "1") != "0",
        use_alias=os.environ.get("RAG_USE_ALIAS", "1") != "0",
        use_ce=os.environ.get("RETRIEVAL_USE_CROSS_ENCODER", "0") == "1",
        use_article_window=os.environ.get("RAG_ARTICLE_WINDOW", "1") != "0",
        use_article_constraint=os.environ.get("RAG_ARTICLE_CONSTRAINT", "1") != "0",
    )
    retriever = build_retriever(vs, cfg)
    return retriever(question, top_k=top_k)


# Deprecated stubs preserved so v2-era diagnostics in evaluate.py do not break.
def classify_intents(question: str) -> list[str]:  # noqa: ARG001
    return []


def expand_traffic_query(question: str) -> str:
    return question


def is_supported_traffic_question(question: str) -> bool:  # noqa: ARG001
    return True
