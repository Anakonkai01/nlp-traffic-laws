"""Query-aware span reranking for legal evidence selection.

This module performs a second-stage ranking over spans split from retrieved
chunks. It is generic: scores combine lexical overlap, optional dense embedding
similarity, and the parent retriever score; it does not map traffic topics to
specific articles or sanctions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from langchain_core.documents import Document

try:
    from context_packing import _split_spans  # type: ignore
except Exception:  # pragma: no cover - fallback keeps the module importable
    _split_spans = None

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _cosine_sparse(a: Counter[str], b: Counter[str], idf: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(a[t] * b.get(t, 0) * (idf.get(t, 1.0) ** 2) for t in a)
    den_a = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in a.items())) or 1.0
    den_b = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in b.items())) or 1.0
    return num / (den_a * den_b)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parent_score(doc: Document, rank: int) -> float:
    md = doc.metadata or {}
    raw = md.get("retrieval_v4_score")
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return 1.0 / (rank + 1)


def _dense_scores(vs, query: str, spans: list[str]) -> list[float]:
    """Return cosine similarities using the vectorstore embedding function when available."""
    emb = getattr(vs, "embedding_function", None)
    if emb is None or not hasattr(emb, "embed_query") or not hasattr(emb, "embed_documents"):
        return [0.0] * len(spans)
    try:
        qv = emb.embed_query(query)
        sv = emb.embed_documents(spans)
    except Exception:
        return [0.0] * len(spans)

    def cos(u, v) -> float:
        num = sum(float(x) * float(y) for x, y in zip(u, v))
        du = math.sqrt(sum(float(x) * float(x) for x in u)) or 1.0
        dv = math.sqrt(sum(float(y) * float(y) for y in v)) or 1.0
        return num / (du * dv)

    return [cos(qv, v) for v in sv]


def rerank_spans(
    docs: Sequence[Document],
    question: str,
    *,
    top_k: int = 5,
    max_chars_per_span: int = 900,
    vs=None,
    use_dense: bool = False,
    w_sparse: float = 0.45,
    w_overlap: float = 0.20,
    w_dense: float = 0.25,
    w_parent: float = 0.10,
) -> list[Document]:
    """Split retrieved chunks into spans and rerank them for the query."""
    if not docs:
        return []
    splitter = _split_spans
    candidates: list[dict] = []
    df: Counter[str] = Counter()
    for doc_rank, doc in enumerate(docs, start=1):
        text = doc.page_content or ""
        spans = splitter(text, max_chars=max_chars_per_span) if splitter else [text[:max_chars_per_span]]
        for span_idx, span in enumerate(spans):
            span = re.sub(r"\s+", " ", span or "").strip()
            if not span:
                continue
            terms = Counter(_tokens(span))
            if not terms:
                continue
            candidates.append({
                "doc": doc,
                "doc_rank": doc_rank,
                "span_idx": span_idx,
                "span": span,
                "terms": terms,
                "parent": _parent_score(doc, doc_rank),
            })
            df.update(terms.keys())
    if not candidates:
        return list(docs)[:top_k]

    q_terms = Counter(_tokens(question))
    q_set = set(q_terms)
    n = len(candidates)
    idf = {term: math.log((n + 1) / (freq + 0.5)) + 1.0 for term, freq in df.items()}
    dense = _dense_scores(vs, question, [c["span"] for c in candidates]) if use_dense else [0.0] * n

    parent_values = [c["parent"] for c in candidates]
    p_min, p_max = min(parent_values), max(parent_values)
    for idx, cand in enumerate(candidates):
        sparse = _cosine_sparse(q_terms, cand["terms"], idf)
        overlap = _jaccard(q_set, set(cand["terms"]))
        parent = 0.0 if p_max == p_min else (cand["parent"] - p_min) / (p_max - p_min)
        score = w_sparse * sparse + w_overlap * overlap + w_dense * dense[idx] + w_parent * parent
        cand["score"] = score
        cand["parts"] = {
            "span_sparse": round(sparse, 6),
            "span_overlap": round(overlap, 6),
            "span_dense": round(dense[idx], 6),
            "parent": round(parent, 6),
        }

    ranked = sorted(candidates, key=lambda c: (-c["score"], c["doc_rank"], c["span_idx"]))[:top_k]
    out: list[Document] = []
    for rank, cand in enumerate(ranked, start=1):
        md = dict(cand["doc"].metadata or {})
        md.update({
            "span_reranked": True,
            "span_rank": rank,
            "span_index": cand["span_idx"],
            "span_score": round(float(cand["score"]), 6),
            "span_score_parts": cand["parts"],
            "context_pack_relation": md.get("context_pack_relation", "span_reranked"),
        })
        out.append(Document(page_content=cand["span"], metadata=md))
    return out
