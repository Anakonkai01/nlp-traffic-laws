"""Compact evidence-card rendering for legal RAG.

The renderer converts raw legal excerpts into short, answer-ready facts. It uses
objective legal structure and regex extraction, not question-specific mappings.
"""
from __future__ import annotations

import re

ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)\.\s*(.*?)(?=\s+\d+\.\s|\n|$)", re.I | re.S)
CLAUSE_RE = re.compile(r"(?s)(?:^|\n|\.\s)(\d+)\.\s+(.+?)(?=(?:\n|\.\s)(?:[a-zđ]\)|\d+\.)|\Z)")
POINT_RE = re.compile(r"(?s)(?:^|\n|\.\s)([a-zđ])\)\s+(.+?)(?=(?:\n|\.\s)[a-zđ]\)|(?:\n|\.\s)\d+\.|\Z)")
FINE_RE = re.compile(
    r"phạt\s+tiền\s+từ\s+([\d\.]+)\s*đồng\s+đến\s+([\d\.]+)\s*đồng|"
    r"phạt\s+tiền\s+([\d\.]+)\s*đồng",
    re.I,
)
POINT_DEDUCT_RE = re.compile(r"trừ\s+(?:điểm\s+giấy\s+phép\s+lái\s+xe\s*)?(\d+)\s+điểm", re.I)
SUSPEND_RE = re.compile(r"tước[^.;\n]{0,120}?(?:từ\s+\d+\s+[^.;\n]+?đến\s+\d+\s+[^.;\n]+|\d+\s+tháng)", re.I)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clip_sentence(text: str, limit: int = 260) -> str:
    text = compact(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + " ..."


def _first(regex: re.Pattern, text: str) -> str:
    m = regex.search(text or "")
    return compact(m.group(0)) if m else ""


def _clean_amount(amount: str) -> str:
    amount = re.sub(r"\s+", "", amount or "")
    return amount.strip(".,;:")


def _fine(text: str) -> str:
    m = FINE_RE.search(text or "")
    if not m:
        return ""
    if m.group(1) and m.group(2):
        lo = _clean_amount(m.group(1))
        hi = _clean_amount(m.group(2))
        return f"phạt tiền từ {lo} đồng đến {hi} đồng"
    return f"phạt tiền {_clean_amount(m.group(3))} đồng"


def _select_point(question: str, text: str) -> tuple[str, str]:
    q_tokens = set(re.findall(r"[\wÀ-ỹ]+", (question or "").lower()))
    best = ("", "", -1)
    for m in POINT_RE.finditer(text or ""):
        body = compact(m.group(2))
        toks = set(re.findall(r"[\wÀ-ỹ]+", body.lower()))
        score = len(q_tokens & toks)
        if score > best[2]:
            best = (m.group(1), body, score)
    return best[0], best[1]


def _select_clause(question: str, text: str) -> tuple[str, str]:
    q_tokens = set(re.findall(r"[\wÀ-ỹ]+", (question or "").lower()))
    best = ("", "", -1)
    for m in CLAUSE_RE.finditer(text or ""):
        body = compact(m.group(0))
        toks = set(re.findall(r"[\wÀ-ỹ]+", body.lower()))
        score = len(q_tokens & toks)
        if _fine(body):
            score += 3
        if score > best[2]:
            best = (m.group(1), body, score)
    return best[0], best[1]


def evidence_card_from_text(question: str, text: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    raw = text or ""
    article_m = ARTICLE_RE.search(raw)
    article_no = str(metadata.get("article_number") or (article_m.group(1) if article_m else ""))
    article_title = compact(metadata.get("article") or (article_m.group(0) if article_m else ""))
    article_title = clip_sentence(article_title, 180)
    clause_no = str(metadata.get("clause_number") or "")
    point_letter = str(metadata.get("point_letter") or "")
    meta_clause_lead = compact(metadata.get("clause_lead") or "")
    meta_point_text = compact(metadata.get("point_text") or "")

    clause_no_auto, clause_text = _select_clause(question, raw)
    point_auto, point_text = _select_point(question, raw)
    if meta_clause_lead:
        clause_text = meta_clause_lead
    if meta_point_text:
        point_text = meta_point_text
    clause_no = clause_no or clause_no_auto
    point_letter = point_letter or point_auto

    fine = _fine(meta_clause_lead) or _fine(clause_text) or _fine(raw)
    deduct_m = POINT_DEDUCT_RE.search(raw or "")
    deduct = f"{deduct_m.group(1)} điểm" if deduct_m else ""
    suspend = _first(SUSPEND_RE, raw)
    violation = clip_sentence(point_text or clause_text or raw, 320)
    doc = metadata.get("doc_id") or metadata.get("source") or ""
    citation_parts = []
    if doc:
        citation_parts.append(str(doc))
    if article_no:
        citation_parts.append(f"Điều {article_no}")
    if clause_no:
        citation_parts.append(f"khoản {clause_no}")
    if point_letter:
        citation_parts.append(f"điểm {point_letter}")

    lines = ["EVIDENCE_CARD"]
    if citation_parts:
        lines.append(f"Căn cứ: {' '.join(citation_parts)}")
    if article_title:
        lines.append(f"Điều luật: {article_title}")
    if fine:
        lines.append(f"Mức phạt: {fine}")
        lines.append(f"Kết luận tiền phạt: {fine}")
    if deduct:
        lines.append(f"Trừ điểm: {deduct}")
    if suspend:
        lines.append(f"Tước/đình chỉ: {suspend}")
    if violation:
        lines.append(f"Hành vi/điều kiện liên quan: {violation}")
    lines.append("Chỉ trả lời bằng kết luận ngắn gọn từ các trường trên; không chép lại thẻ.")
    return "\n".join(lines)
