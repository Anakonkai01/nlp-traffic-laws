"""Structure-aware legal-unit extraction and retrieval.

This module indexes objective legal structure (document -> article -> clause ->
point) rather than hand-written traffic rules. It is intended as a second-stage
candidate generator after coarse retrieval has found the right source/article.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document

from config import DATA_DIR
from corpus import compact_text, iter_traffic_documents
from sparse_bm25 import tokenize

LEGAL_UNITS_PATH = DATA_DIR / "legal_units.jsonl"
ARTICLE_RE = re.compile(r"(?m)^Điều\s+(\d+[a-zA-Z]?)\.\s*[^\n]*")
CLAUSE_RE = re.compile(r"(?m)^(\d+)\.\s+")
POINT_RE = re.compile(r"(?m)^([a-zđ])\)\s+")
VEHICLE_ALIASES = {
    "special_motorbike": ["xe máy chuyên dùng"],
    "motorbike": ["xe mô tô", "mô tô", "xe gắn máy", "xe máy"],
    "car": ["xe ô tô", "ô tô", "xe hơi"],
    "bicycle": ["xe đạp máy", "xe đạp"],
    "pedestrian": ["người đi bộ"],
}
SANCTION_TERMS = {
    "deduct_points": ["trừ điểm", "điểm giấy phép"],
    "suspension": ["tước", "giấy phép lái xe"],
    "fine": ["phạt tiền", "mức phạt", "bao nhiêu tiền"],
}


def _unit_id(*parts: str | None) -> str:
    return "::".join(str(p or "_") for p in parts)


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in (text or "").lower()


def vehicle_profile(text: str) -> set[str]:
    low = (text or "").lower()
    found: set[str] = set()
    # Longest/specific aliases first so "xe máy chuyên dùng" is not treated as normal xe máy.
    for label, aliases in VEHICLE_ALIASES.items():
        if any(alias in low for alias in aliases):
            found.add(label)
    if "special_motorbike" in found and not any(x in low for x in ["mô tô", "xe mô tô", "xe gắn máy"]):
        found.discard("motorbike")
    return found


def sanction_profile(text: str) -> set[str]:
    low = (text or "").lower()
    return {label for label, aliases in SANCTION_TERMS.items() if any(alias in low for alias in aliases)}


def _entity_adjustment(question: str, doc: Document) -> tuple[float, dict]:
    md = doc.metadata or {}
    candidate_text = f"{md.get('article', '')} {md.get('legal_path', '')} {doc.page_content or ''}"
    qv = vehicle_profile(question)
    cv = vehicle_profile(candidate_text)
    qs = sanction_profile(question)
    cs = sanction_profile(candidate_text)
    adj = 0.0
    details = {"query_vehicle": sorted(qv), "candidate_vehicle": sorted(cv), "query_sanction": sorted(qs), "candidate_sanction": sorted(cs)}
    if qv and cv:
        if qv & cv:
            adj += 0.8
        elif qv == {"motorbike"} and cv == {"special_motorbike"}:
            adj -= 2.0
        else:
            adj -= 1.2
    if qs and cs:
        adj += 0.25 * len(qs & cs)
    details["entity_adjustment"] = round(adj, 4)
    return adj, details


def _article_spans(text: str):
    matches = list(ARTICLE_RE.finditer(text))
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield match, text[match.start():end].strip()


def _clause_spans(article_text: str):
    matches = list(CLAUSE_RE.finditer(article_text))
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(article_text)
        yield match, article_text[match.start():end].strip()


def _point_spans(clause_text: str):
    matches = list(POINT_RE.finditer(clause_text))
    lead = clause_text[: matches[0].start()].strip() if matches else ""
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clause_text)
        yield match, lead, clause_text[match.start():end].strip()


def extract_legal_units(docs: Iterable[Document] | None = None) -> list[Document]:
    """Extract article/clause/point units with legal path metadata."""
    units: list[Document] = []
    for doc in docs or iter_traffic_documents():
        md = dict(doc.metadata or {})
        doc_id = md.get("doc_id") or md.get("source") or "unknown"
        title = md.get("title") or ""
        text = doc.page_content or ""
        for article_match, article_text in _article_spans(text):
            article_no = article_match.group(1)
            article_title = compact_text(article_match.group(0))
            article_md = {
                **md,
                "unit_type": "article",
                "article": article_title,
                "article_number": article_no,
                "legal_path": f"{title} > {article_title}",
            }
            units.append(Document(page_content=article_text, metadata={**article_md, "unit_id": _unit_id(doc_id, "article", article_no)}))

            clauses = list(_clause_spans(article_text))
            if not clauses:
                continue
            for clause_match, clause_text in clauses:
                clause_no = clause_match.group(1)
                clause_header = f"{article_title}\nKhoản {clause_no}"
                clause_md = {
                    **article_md,
                    "unit_type": "clause",
                    "clause_number": clause_no,
                    "legal_path": f"{title} > {article_title} > khoản {clause_no}",
                }
                units.append(
                    Document(
                        page_content=f"{clause_header}\n\n{clause_text}",
                        metadata={**clause_md, "unit_id": _unit_id(doc_id, "article", article_no, "clause", clause_no)},
                    )
                )

                points = list(_point_spans(clause_text))
                for point_match, clause_lead, point_text in points:
                    point = point_match.group(1)
                    point_md = {
                        **clause_md,
                        "unit_type": "point",
                        "point_letter": point,
                        "clause_lead": compact_text(clause_lead),
                        "point_text": compact_text(point_text),
                        "legal_path": f"{title} > {article_title} > khoản {clause_no} > điểm {point}",
                    }
                    units.append(
                        Document(
                            page_content=f"{article_title}\nKhoản {clause_no} điểm {point}\n\n{clause_lead}\n{point_text}".strip(),
                            metadata={**point_md, "unit_id": _unit_id(doc_id, "article", article_no, "clause", clause_no, "point", point)},
                        )
                    )
    return units


def save_legal_units(path: Path = LEGAL_UNITS_PATH) -> list[Document]:
    units = extract_legal_units()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in units:
            f.write(json.dumps({"text": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")
    return units


def load_legal_units(path: Path = LEGAL_UNITS_PATH) -> list[Document]:
    if not path.exists():
        return save_legal_units(path)
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row.get("metadata") or {}))
    return docs


@dataclass
class LegalUnitBM25:
    docs: list[Document]
    tokenized: list[list[str]]
    bm25: object

    @classmethod
    def build(cls, docs: Sequence[Document]) -> "LegalUnitBM25":
        from rank_bm25 import BM25Okapi
        # Index path metadata with body text so exact article/clause mentions help ranking.
        tokenized = [tokenize(f"{(d.metadata or {}).get('legal_path', '')}\n{d.page_content or ''}") for d in docs]
        return cls(list(docs), tokenized, BM25Okapi(tokenized))

    def search(self, query: str, k: int = 50, candidates: Sequence[Document] | None = None) -> list[tuple[Document, float]]:
        import numpy as np

        allowed = None
        if candidates is not None:
            allowed = {str((d.metadata or {}).get("unit_id")) for d in candidates}
        scores = self.bm25.get_scores(tokenize(query))
        if allowed is not None:
            for idx, doc in enumerate(self.docs):
                if str((doc.metadata or {}).get("unit_id")) not in allowed:
                    scores[idx] = -1e9
        k = min(k, len(scores))
        order = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else range(len(scores))
        order = sorted(order, key=lambda i: -scores[i])
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > -1e8]


def candidate_units_from_seeds(
    units: Sequence[Document],
    seeds: Sequence[Document],
    *,
    max_units: int = 80,
    allowed_types: set[str] | None = None,
) -> list[Document]:
    """Expand objective child units from retrieved source/article seeds.

    Collect all matching units before capping; capping during corpus-order scans can
    hide later clauses/points in long sanction articles before BM25 sees them.
    """
    keys = {
        (str((s.metadata or {}).get("doc_id") or (s.metadata or {}).get("source") or ""), str((s.metadata or {}).get("article_number") or ""))
        for s in seeds
    }
    out: list[Document] = []
    seen: set[str] = set()
    for unit in units:
        md = unit.metadata or {}
        if allowed_types and str(md.get("unit_type") or "") not in allowed_types:
            continue
        key = (str(md.get("doc_id") or md.get("source") or ""), str(md.get("article_number") or ""))
        uid = str(md.get("unit_id"))
        if key in keys and uid not in seen:
            seen.add(uid)
            out.append(unit)
    return out


_CACHE: dict[str, object] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _ce_model():
    model_id = os.environ.get("RAG_LEGAL_UNIT_CE_MODEL") or "models/bge-reranker-v2-m3-traffic-ft"
    key = f"ce::{model_id}"
    if key not in _CACHE:
        from sentence_transformers import CrossEncoder
        _CACHE[key] = CrossEncoder(model_id, max_length=int(os.environ.get("RAG_LEGAL_UNIT_CE_MAX_LENGTH", "512")))
    return _CACHE[key]


def retrieve_legal_units(question: str, seeds: Sequence[Document], *, top_k: int = 8, max_candidates: int = 80) -> list[Document]:
    units = _CACHE.get("units")
    if units is None:
        units = load_legal_units()
        _CACHE["units"] = units
        _CACHE["bm25"] = LegalUnitBM25.build(units)  # type: ignore[arg-type]
    bm25 = _CACHE["bm25"]
    allowed = {x.strip() for x in os.environ.get("RAG_LEGAL_UNIT_ALLOWED_TYPES", "point,clause").split(",") if x.strip()}
    candidates = candidate_units_from_seeds(units, seeds, max_units=max_candidates, allowed_types=allowed)  # type: ignore[arg-type]
    if not candidates:
        candidates = units  # type: ignore[assignment]
    bm25_k = max(top_k, int(os.environ.get("RAG_LEGAL_UNIT_BM25_CANDIDATES", "30")))
    ranked = bm25.search(question, k=min(bm25_k, len(candidates)), candidates=candidates)  # type: ignore[attr-defined]
    if _env_bool("RAG_LEGAL_UNIT_ENTITY_SCORING", True) and ranked:
        adjusted = []
        for doc, score in ranked:
            adj, detail = _entity_adjustment(question, doc)
            adjusted.append((doc, score + adj, score, detail))
        adjusted.sort(key=lambda item: -item[1])
        ranked = []
        for doc, adjusted_score, bm25_score, detail in adjusted:
            md = dict(doc.metadata or {})
            md["legal_unit_entity_features"] = detail
            md["legal_unit_adjusted_score"] = adjusted_score
            ranked.append((Document(page_content=doc.page_content, metadata=md), bm25_score))

    if _env_bool("RAG_LEGAL_UNIT_USE_CE", False) and ranked:
        ce = _ce_model()
        docs = [doc for doc, _ in ranked]
        scores = ce.predict([(question, doc.page_content or "") for doc in docs], batch_size=32, show_progress_bar=False)
        ranked_docs = sorted(zip(docs, ranked, scores), key=lambda item: -float(item[2]))[:top_k]
        out = []
        for rank, (doc, (_orig_doc, bm25_score), ce_score) in enumerate(ranked_docs, start=1):
            md = dict(doc.metadata or {})
            md["legal_unit_bm25_score"] = bm25_score
            md["legal_unit_ce_score"] = float(ce_score)
            md["legal_unit_rank"] = rank
            out.append(Document(page_content=doc.page_content, metadata=md))
        return out

    out: list[Document] = []
    for rank, (doc, score) in enumerate(ranked[:top_k], start=1):
        md = dict(doc.metadata or {})
        md["legal_unit_bm25_score"] = score
        md["legal_unit_rank"] = rank
        out.append(Document(page_content=doc.page_content, metadata=md))
    return out
