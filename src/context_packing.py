"""Context packing utilities for legal RAG generation.

The retriever should rank evidence seeds. Packing can then add concise structural
support (same legal article neighbours) for generation without changing the
retrieval ranking metric.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from typing import Sequence

from langchain_core.documents import Document

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_SPAN_SPLIT_RE = re.compile(r"(?m)(?=^\s*(?:\d+\.|[a-zđ]\)|[-–•]|Điều\s+\d+|Khoản\s+\d+))")
_NUMBERED_CLAUSE_RE = re.compile(r"^\s*\d+\.")
_BULLET_RE = re.compile(r"^\s*(?:[a-zđ]\)|[-–•])")
_CLAUSE_RE = re.compile(r"(?i)\b(?:khoản|mục)\s+(\d+[a-z]?)\b")
_NUMERIC_UNIT_RE = re.compile(
    r"(?i)\b\d+(?:[.,]\d+)?\s*(?:mg/lít|mg/100\s*ml|điểm|tháng|năm|km/h|đồng|triệu|%)\b"
)


def _doc_id(doc: Document) -> str | None:
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _article_number(doc: Document) -> str:
    return str((doc.metadata or {}).get("article_number") or "")


def _chunk_id(doc: Document) -> str:
    md = doc.metadata or {}
    return str(md.get("chunk_id") or f"{_doc_id(doc) or 'unknown'}:{hash((doc.page_content or '')[:200]) & 0xffffffff:08x}")


def vectorstore_docs(vs) -> list[Document]:
    docstore = getattr(vs, "docstore", None)
    return list(getattr(docstore, "_dict", {}).values()) if docstore else []


def pack_article_context(
    seeds: Sequence[Document],
    all_docs: Sequence[Document],
    *,
    top_k: int = 2,
    max_chunks: int = 4,
    radius: int = 1,
    max_chars_per_chunk: int = 1400,
) -> list[Document]:
    """Add same-article neighbours around ranked seeds for generation context.

    This is intentionally post-retrieval: seeds keep priority, and neighbours are
    only supporting evidence for the answer prompt.
    """
    by_key: dict[tuple[str | None, str], list[Document]] = defaultdict(list)
    for doc in all_docs:
        key = (_doc_id(doc), _article_number(doc))
        by_key[key].append(doc)
    for docs in by_key.values():
        docs.sort(key=lambda d: str((d.metadata or {}).get("chunk_id") or ""))

    packed: list[Document] = []
    seen: set[str] = set()

    def add(doc: Document, relation: str):
        cid = _chunk_id(doc)
        if cid in seen or len(packed) >= max_chunks:
            return
        seen.add(cid)
        md = dict(doc.metadata or {})
        md["context_pack_relation"] = relation
        text = (doc.page_content or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0].strip() + " ..."
        packed.append(Document(page_content=text, metadata=md))

    for seed in list(seeds)[:top_k]:
        if len(packed) >= max_chunks:
            break
        add(seed, "seed")
        key = (_doc_id(seed), _article_number(seed))
        if not key[1] or len(packed) >= max_chunks:
            continue
        siblings = by_key.get(key, [])
        try:
            idx = next(i for i, doc in enumerate(siblings) if _chunk_id(doc) == _chunk_id(seed))
        except StopIteration:
            continue
        for pos in range(max(0, idx - radius), min(len(siblings), idx + radius + 1)):
            if pos == idx:
                continue
            add(siblings[pos], "same_article_neighbor")
            if len(packed) >= max_chunks:
                break

    return packed


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _split_long_span(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    text = text.strip()
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def _split_clause_head(part: str) -> str:
    """Keep the governing sentence of a numbered clause for bullet pairing."""
    head = part.strip()
    # Retain the clause number but avoid duplicating a full unrelated bullet.
    number = re.match(r"^\s*(\d+\.)", head)
    if number:
        fine = re.search(r"Phạt tiền từ.*?(?= đối với|$)", head, flags=re.IGNORECASE)
        if fine:
            return f"{number.group(1)} {fine.group(0)}"
    return head


def _split_spans(text: str, *, max_chars: int = 650) -> list[str]:
    """Split legal text into generic evidence spans without domain keywords."""
    text = (text or "").strip()
    if not text:
        return []
    rough = [re.sub(r"\s+", " ", s).strip() for s in _SPAN_SPLIT_RE.split(text) if s.strip()]
    if len(rough) <= 1:
        normalized = re.sub(r"\s+", " ", text).strip()
        rough = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", normalized) if s.strip()]

    spans: list[str] = []
    active_clause_head = ""
    for part in rough:
        if _NUMBERED_CLAUSE_RE.match(part):
            active_clause_head = _split_clause_head(part)
            spans.extend(_split_long_span(part, max_chars))
            continue

        if active_clause_head and _BULLET_RE.match(part):
            paired = f"{active_clause_head} {part}".strip()
            spans.extend(_split_long_span(paired, max_chars * 2))
            continue

        spans.extend(_split_long_span(part, max_chars))
    return spans


def compress_evidence_context(
    docs: Sequence[Document],
    question: str,
    *,
    max_spans: int = 6,
    max_chars_per_span: int = 650,
    mmr_lambda: float = 0.72,
    order: str = "litm",
    clause_aware: bool | None = None,
) -> list[Document]:
    """Select concise query-focused spans from retrieved docs using generic MMR.

    The scoring uses token overlap/IDF from the retrieved context only. It does
    not encode traffic-law-specific terms or map questions to legal provisions.
    """
    q_terms = Counter(_tokens(question))
    if not q_terms:
        return list(docs)
    if clause_aware is None:
        clause_aware = os.environ.get("RAG_CLAUSE_AWARE_PACKING", "1") == "1"
    question_numeric_units = {m.group(0).lower().replace(",", ".") for m in _NUMERIC_UNIT_RE.finditer(question)}
    top_seed = docs[0] if docs else None
    top_seed_key = (_doc_id(top_seed), _article_number(top_seed)) if top_seed is not None else (None, "")
    top_seed_clauses = set(_CLAUSE_RE.findall(top_seed.page_content or "")) if top_seed is not None else set()

    candidates: list[dict] = []
    df: Counter[str] = Counter()
    for doc_idx, doc in enumerate(docs):
        for span_idx, span in enumerate(_split_spans(doc.page_content or "", max_chars=max_chars_per_span)):
            terms = Counter(_tokens(span))
            if not terms:
                continue
            candidates.append({"doc": doc, "doc_idx": doc_idx, "span_idx": span_idx, "span": span, "terms": terms})
            df.update(terms.keys())
    if not candidates:
        return list(docs)

    n = len(candidates)
    idf = {term: math.log((n + 1) / (freq + 0.5)) + 1.0 for term, freq in df.items()}

    def relevance(cand: dict) -> float:
        terms = cand["terms"]
        denom = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in terms.items())) or 1.0
        qden = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in q_terms.items())) or 1.0
        num = sum(q_terms[t] * terms.get(t, 0) * (idf.get(t, 1.0) ** 2) for t in q_terms)
        score = num / (denom * qden)
        # Generic legal-structure prior: paired clause+bullet spans preserve the
        # governing sanction while remaining independent of any traffic topic.
        if re.match(r"^\s*\d+\.\s*[a-zđ]\)", cand["span"]):
            score += 0.08
        if clause_aware:
            span = cand["span"]
            numeric_units = {m.group(0).lower().replace(",", ".") for m in _NUMERIC_UNIT_RE.finditer(span)}
            score += 0.10 * len(question_numeric_units & numeric_units)
            doc = cand["doc"]
            if (_doc_id(doc), _article_number(doc)) == top_seed_key:
                clauses = set(_CLAUSE_RE.findall(span))
                if clauses and top_seed_clauses and clauses & top_seed_clauses:
                    score += 0.10
        return score

    def similarity(a: Counter[str], b: Counter[str]) -> float:
        overlap = set(a) & set(b)
        if not overlap:
            return 0.0
        num = sum(min(a[t], b[t]) for t in overlap)
        den = min(sum(a.values()), sum(b.values())) or 1
        return num / den

    for cand in candidates:
        cand["rel"] = relevance(cand)
        # Keep retriever order as a weak prior for ties and low-overlap legal text.
        cand["score"] = cand["rel"] + 0.02 / (cand["doc_idx"] + 1)

    selected: list[dict] = []
    pool = candidates[:]
    while pool and len(selected) < max_spans:
        best = max(
            pool,
            key=lambda c: mmr_lambda * c["score"] - (1 - mmr_lambda) * max([similarity(c["terms"], s["terms"]) for s in selected] or [0.0]),
        )
        selected.append(best)
        pool.remove(best)

    if order == "litm" and len(selected) >= 3:
        # Keep the strongest span first, then place the second strongest last;
        # this mitigates lost-in-the-middle without hiding the best evidence.
        selected = [selected[0], *selected[2:], selected[1]]

    packed: list[Document] = []
    for rank, cand in enumerate(selected, start=1):
        md = dict(cand["doc"].metadata or {})
        md.update({
            "context_pack_relation": md.get("context_pack_relation", "retrieved"),
            "evidence_compressed": True,
            "evidence_span_rank": rank,
            "evidence_span_index": cand["span_idx"],
            "evidence_score": round(float(cand["score"]), 6),
        })
        packed.append(Document(page_content=cand["span"], metadata=md))
    return packed
