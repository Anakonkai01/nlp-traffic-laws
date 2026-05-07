"""Structured sanction fact retrieval and card rendering."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from config import DATA_DIR
from sparse_bm25 import tokenize

# Map English vehicle labels (stored in legal_sanction_facts) to Vietnamese terms used in queries.
_VEHICLE_VI: dict[str, str] = {
    "car": "ô tô xe hơi xe bốn bánh",
    "motorbike": "xe máy xe mô tô gắn máy mô tô",
    "bicycle": "xe đạp",
    "truck": "xe tải xe chở hàng",
    "bus": "xe buýt xe khách xe ô tô khách",
}

FACTS_PATH = DATA_DIR / "legal_sanction_facts.jsonl"


def load_sanction_facts(path: Path = FACTS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sanction facts: {path}. Run scripts/build_sanction_facts.py")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def fact_to_text(f: dict) -> str:
    parts = [
        f.get("citation", ""),
        f.get("article_title", ""),
        " ".join(f.get("vehicle_scope") or []),
        f.get("violation_text", ""),
        f.get("fact_type", ""),
        "answer_ready" if f.get("answer_ready") else "",
        f.get("fine_text", ""),
        f.get("points_deducted", "") or "",
        f.get("suspension_text", "") or "",
    ]
    return "\n".join(str(p) for p in parts if p)


def _tokens(text: str) -> set[str]:
    return {t for t in tokenize(text or "") if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _amount_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d{1,3}(?:\.\d{3})+|\d+", text or ""))


def _fact_score(question: str, fact: dict, bm25_score: float) -> tuple[float, dict]:
    q_tok = _tokens(question)
    v_tok = _tokens(fact.get("violation_text", ""))
    title_tok = _tokens(fact.get("article_title", ""))
    fine_tok = _tokens(fact.get("fine_text", ""))
    vehicle_vi = " ".join(_VEHICLE_VI.get(v, v) for v in (fact.get("vehicle_scope") or []))
    vehicle_tok = _tokens(vehicle_vi)
    q_amounts = _amount_tokens(question)
    f_amounts = _amount_tokens(" ".join(str(fact.get(k) or "") for k in ["fine_text", "points_deducted", "suspension_text"]))
    bm25_norm = bm25_score / (bm25_score + 8.0) if bm25_score > 0 else 0.0
    comps = {
        "bm25_norm": bm25_norm,
        "violation_overlap": _jaccard(q_tok, v_tok),
        "title_overlap": _jaccard(q_tok, title_tok),
        "fine_overlap": _jaccard(q_tok, fine_tok),
        "vehicle_overlap": _jaccard(q_tok, vehicle_tok),
        "amount_overlap": len(q_amounts & f_amounts) / max(1, len(q_amounts)) if q_amounts else 0.0,
        "has_answer_field": 1.0 if fact.get("fine_text") or fact.get("points_deducted") or fact.get("suspension_text") else 0.0,
        "is_sanction": 1.0 if fact.get("fact_type") in {"sanction", "deduction", "suspension"} else 0.0,
    }
    score = (
        0.42 * comps["bm25_norm"]
        + 0.30 * comps["violation_overlap"]
        + 0.08 * comps["title_overlap"]
        + 0.12 * comps["vehicle_overlap"]
        + 0.04 * comps["fine_overlap"]
        + 0.02 * comps["amount_overlap"]
        + 0.015 * comps["has_answer_field"]
        + 0.005 * comps["is_sanction"]
    )
    return score, comps


def fact_to_card(f: dict) -> str:
    fmt = os.environ.get("RAG_FACT_CARD_FORMAT", "evidence").lower()
    if fmt in {"narrative", "plain"}:
        bits = []
        if f.get("citation"):
            bits.append(f"Theo {f['citation']}")
        if f.get("vehicle_scope"):
            bits.append(f"áp dụng cho {', '.join(f['vehicle_scope'])}")
        if f.get("violation_text"):
            bits.append(f"hành vi {f['violation_text'][:300]}")
        if f.get("fine_text"):
            bits.append(f"bị {f['fine_text']}")
        if f.get("points_deducted"):
            bits.append(f"và bị trừ {f['points_deducted']} giấy phép lái xe")
        if f.get("suspension_text"):
            bits.append(f"; hình thức bổ sung: {f['suspension_text']}")
        sentence = ", ".join(bits).strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        return sentence

    concise = fmt in {"answer", "concise"}
    if concise:
        fields = ["ANSWER_FACT"]
        if f.get("fine_text"):
            fields.append(f"Mức phạt: {f['fine_text']}")
        if f.get("points_deducted"):
            fields.append(f"Trừ điểm: {f['points_deducted']}")
        if f.get("suspension_text"):
            fields.append(f"Tước/đình chỉ: {f['suspension_text']}")
        if f.get("citation"):
            fields.append(f"Căn cứ: {f['citation']}")
        if f.get("vehicle_scope"):
            fields.append(f"Phương tiện: {', '.join(f['vehicle_scope'])}")
        if f.get("violation_text"):
            fields.append(f"Hành vi: {f['violation_text'][:320]}")
        fields.append("Trả lời trực tiếp bằng Mức phạt/Trừ điểm/Tước nếu có; không chép nhãn ANSWER_FACT.")
        return "\n".join(fields)

    lines = ["EVIDENCE_CARD"]
    if f.get("citation"):
        lines.append(f"Căn cứ: {f['citation']}")
    if f.get("article_title"):
        lines.append(f"Điều luật: {f['article_title']}")
    if f.get("vehicle_scope"):
        lines.append(f"Phương tiện/đối tượng: {', '.join(f['vehicle_scope'])}")
    if f.get("violation_text"):
        lines.append(f"Hành vi/điều kiện liên quan: {f['violation_text'][:420]}")
    if f.get("fine_text"):
        lines.append(f"Mức phạt: {f['fine_text']}")
        lines.append(f"Kết luận tiền phạt: {f['fine_text']}")
    if f.get("points_deducted"):
        lines.append(f"Trừ điểm: {f['points_deducted']}")
    if f.get("suspension_text"):
        lines.append(f"Tước/đình chỉ: {f['suspension_text']}")
    lines.append("Chỉ trả lời bằng kết luận ngắn gọn từ các trường trên; không chép lại thẻ.")
    return "\n".join(lines)


@dataclass
class FactBM25:
    facts: list[dict]
    tokenized: list[list[str]]
    bm25: object

    @classmethod
    def build(cls, facts: Sequence[dict]) -> "FactBM25":
        from rank_bm25 import BM25Okapi
        tokenized = [tokenize(fact_to_text(f)) for f in facts]
        return cls(list(facts), tokenized, BM25Okapi(tokenized))

    def search(self, query: str, candidates: Sequence[dict], k: int = 3) -> list[tuple[dict, float]]:
        import numpy as np
        use_rerank = os.environ.get("RAG_FACT_GENERIC_RERANK", "1") == "1"
        allowed = {f.get("fact_id") for f in candidates}
        scores = self.bm25.get_scores(tokenize(query))
        for idx, f in enumerate(self.facts):
            if f.get("fact_id") not in allowed:
                scores[idx] = -1e9
        pool_k = min(max(k * 8, 30), len(scores)) if use_rerank else min(k, len(scores))
        order = np.argpartition(-scores, pool_k - 1)[:pool_k] if pool_k < len(scores) else range(len(scores))
        if use_rerank:
            rescored = []
            for i in order:
                if scores[i] <= -1e8:
                    continue
                score, comps = _fact_score(query, self.facts[i], float(scores[i]))
                self.facts[i]["_rerank_score"] = score
                self.facts[i]["_rerank_components"] = comps
                rescored.append((i, score))
            order = [i for i, _ in sorted(rescored, key=lambda x: -x[1])[:k]]
            return [(self.facts[i], float(self.facts[i].get("_rerank_score", scores[i]))) for i in order]
        order = sorted(order, key=lambda i: -scores[i])[:k]
        return [(self.facts[i], float(scores[i])) for i in order if scores[i] > -1e8]


_CACHE: dict[str, object] = {}


def _ce_model():
    if os.environ.get("RAG_FACT_USE_CE", "0") != "1":
        return None
    if "ce" not in _CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = os.environ.get("RAG_FACT_CE_MODEL", "models/fact-reranker-v1")
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        _CACHE["ce"] = (tok, mdl, device)
    return _CACHE["ce"]


def _ce_rerank(question: str, ranked: list[tuple[dict, float]], k: int) -> list[tuple[dict, float]]:
    ce = _ce_model()
    if ce is None or not ranked:
        return ranked[:k]
    import torch
    tok, mdl, device = ce
    texts = [fact_to_text(f) for f, _ in ranked]
    scores = []
    bs = int(os.environ.get("RAG_FACT_CE_BS", "8"))
    with torch.no_grad():
        for start in range(0, len(texts), bs):
            enc = tok([question] * len(texts[start:start+bs]), texts[start:start+bs], padding=True, truncation=True, max_length=int(os.environ.get("RAG_FACT_CE_MAX_LENGTH", "384")), return_tensors="pt").to(device)
            logits = mdl(**enc).logits.squeeze(-1).float().detach().cpu().tolist()
            if isinstance(logits, float): logits = [logits]
            scores.extend(logits)
    out = []
    for (fact, old), s in zip(ranked, scores):
        fact["_ce_score"] = float(s)
        out.append((fact, float(s)))
    return sorted(out, key=lambda x: -x[1])[:k]


def _facts_and_bm25() -> tuple[list[dict], FactBM25]:
    if "facts" not in _CACHE:
        facts = load_sanction_facts()
        _CACHE["facts"] = facts
        _CACHE["bm25"] = FactBM25.build(facts)
    return _CACHE["facts"], _CACHE["bm25"]  # type: ignore[return-value]


def _seed_keys(seeds: Sequence[Document]) -> set[tuple[str, str]]:
    return {
        (str((s.metadata or {}).get("doc_id") or (s.metadata or {}).get("source") or ""), str((s.metadata or {}).get("article_number") or ""))
        for s in seeds
    }


def candidate_facts_from_seeds(facts: Sequence[dict], seeds: Sequence[Document]) -> list[dict]:
    keys = _seed_keys(seeds)
    out = [f for f in facts if (str(f.get("doc_id") or ""), str(f.get("article_number") or "")) in keys]
    # Prefer answer-bearing facts but retain broader same-article facts as fallback.
    answerish = [f for f in out if f.get("answer_ready") or f.get("fine_text") or f.get("points_deducted") or f.get("suspension_text")]
    return answerish or out


def _expand_violation_terms(question: str) -> str:
    """Expand only violation-specific colloquial→legal mappings for fact BM25.

    Unlike expand_traffic_query(), this deliberately skips generic vehicle
    expansions ("xe máy" → "xe mô tô...") which appear in every fact and
    dilute the signal for violation-specific retrieval.
    """
    from retrieval import INTENT_PHRASES, QUERY_TO_LEGAL, classify_intents, _normalize
    q_norm = _normalize(question)
    intents = classify_intents(question)
    expansions: list[str] = []
    # Add only violation-type intent phrases (not vehicle/procedural ones)
    for intent in ("signal", "helmet", "alcohol", "license_points"):
        if intent in intents and intent in INTENT_PHRASES:
            expansions.append(INTENT_PHRASES[intent])
    # Add violation-specific surface→legal mappings; skip vehicle-type entries
    _vehicle_surfaces = {"xe hơi", "xe máy", "mô tô", "xe đạp"}
    for surface, legal in QUERY_TO_LEGAL.items():
        if surface in _vehicle_surfaces:
            continue
        if surface in q_norm and legal not in "\n".join(expansions):
            expansions.append(legal)
    if not expansions:
        return question
    return question + "\n" + "\n".join(dict.fromkeys(expansions))


def retrieve_fact_cards(question: str, seeds: Sequence[Document], *, top_k: int = 3) -> list[Document]:
    facts, bm25 = _facts_and_bm25()
    candidates = candidate_facts_from_seeds(facts, seeds)
    if not candidates:
        candidates = facts
    ce_pool_k = int(os.environ.get("RAG_FACT_CE_POOL_K", "20")) if os.environ.get("RAG_FACT_USE_CE", "0") == "1" else top_k
    # Expand only violation terms so colloquial words ("bia") map to legal phrases ("nồng độ cồn")
    # without adding generic vehicle terms that dilute drug/helmet/etc. signals.
    expanded = _expand_violation_terms(question)
    ranked = bm25.search(expanded, candidates, k=max(top_k, ce_pool_k))
    ranked = _ce_rerank(question, ranked, top_k)
    docs: list[Document] = []
    for rank, (fact, score) in enumerate(ranked, start=1):
        md = {
            "doc_id": fact.get("doc_id", ""),
            "source": fact.get("doc_id", ""),
            "article": fact.get("article_title", ""),
            "article_number": fact.get("article_number", ""),
            "clause_number": fact.get("clause_number", ""),
            "point_letter": fact.get("point_letter", ""),
            "unit_id": fact.get("fact_id", ""),
            "unit_type": f"fact:{fact.get('unit_type', '')}",
            "fact_type": fact.get("fact_type", ""),
            "answer_ready": fact.get("answer_ready", False),
            "legal_path": fact.get("citation", ""),
            "fact_score": score,
            "fact_ce_score": fact.get("_ce_score"),
            "fact_rerank_components": fact.get("_rerank_components", {}),
            "fact_rank": rank,
            "source_path": fact.get("source_path", ""),
        }
        docs.append(Document(page_content=fact_to_card(fact), metadata=md))
    return docs
