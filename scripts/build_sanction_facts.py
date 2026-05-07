#!/usr/bin/env python
"""Build structured sanction facts from legal units.

This is a statute-structure parser, not a question-to-law rulebase. It creates
self-contained fact rows by combining point text with parent clause lead and
same-article deduction/suspension references.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from legal_units import load_legal_units, vehicle_profile  # noqa: E402

FINE_RANGE_RE = re.compile(r"phạt\s+tiền\s+từ\s+([\d\.\s]+)\s*(?:đồng)?\s+đến\s+([\d\.\s]+)\s*đồng", re.I)
POINTS_RE = re.compile(r"trừ\s+(?:điểm\s+giấy\s+phép\s+lái\s+xe\s*)?(\d+)\s+điểm", re.I)
SUSPEND_RE = re.compile(r"tước[^.;\n]{0,160}?(?:từ\s+\d+\s+[^.;\n]+?đến\s+\d+\s+[^.;\n]+|\d+\s+tháng)", re.I)
REF_POINT_RE = re.compile(r"điểm\s+([a-zđ](?:\s*,\s*điểm\s+[a-zđ])*)\s+khoản\s+(\d+)", re.I)
POINT_TOKEN_RE = re.compile(r"[a-zđ]")
SANCTION_TITLE_RE = re.compile(r"xử\s+phạt|vi\s+phạm|hình\s+thức|mức\s+phạt", re.I)
PROCEDURE_TITLE_RE = re.compile(r"thủ\s+tục|thẩm\s+quyền|biên\s+bản|quản\s+lý|phục\s+hồi|đối\s+tượng\s+áp\s+dụng|phạm\s+vi", re.I)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def amount_to_int(s: str) -> int | None:
    digits = re.sub(r"\D", "", s or "")
    return int(digits) if digits else None


def amount_display(n: int | None) -> str:
    return f"{n:,}".replace(",", ".") if n is not None else ""


def fine_from_text(text: str) -> tuple[int | None, int | None, str]:
    m = FINE_RANGE_RE.search(text or "")
    if not m:
        return None, None, ""
    lo = amount_to_int(m.group(1)); hi = amount_to_int(m.group(2))
    if lo is None or hi is None:
        return lo, hi, compact(m.group(0))
    return lo, hi, f"phạt tiền từ {amount_display(lo)} đồng đến {amount_display(hi)} đồng"


def classify_fact_type(article_title: str, text: str, fine_text: str, points: str | None, suspension: str | None) -> str:
    joined = f"{article_title} {text}"
    low = joined.lower()
    if fine_text:
        return "sanction"
    if points:
        return "deduction"
    if suspension:
        return "suspension"
    if SANCTION_TITLE_RE.search(joined):
        return "sanction_context"
    if PROCEDURE_TITLE_RE.search(joined):
        return "procedure"
    if "giải thích từ ngữ" in low or "được hiểu" in low:
        return "definition"
    return "context"


def point_refs(text: str) -> list[tuple[str, str]]:
    refs = []
    for m in REF_POINT_RE.finditer(text or ""):
        points = POINT_TOKEN_RE.findall(m.group(1).lower())
        clause = m.group(2)
        refs.extend((clause, p) for p in points)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/legal_units.jsonl")
    ap.add_argument("--output", default="data/legal_sanction_facts.jsonl")
    args = ap.parse_args()

    units = load_legal_units(Path(args.units))
    by_article = defaultdict(list)
    for u in units:
        md = u.metadata or {}
        by_article[(md.get("doc_id") or md.get("source") or "", str(md.get("article_number") or ""))].append(u)

    deduction = {}
    suspension = {}
    for key, article_units in by_article.items():
        for u in article_units:
            text = u.page_content or ""
            if "trừ" in text.lower() and "điểm" in text.lower():
                pts = POINTS_RE.search(text)
                if pts:
                    for clause, point in point_refs(text):
                        deduction[(key[0], key[1], clause, point)] = f"{pts.group(1)} điểm"
            if "tước" in text.lower():
                sm = SUSPEND_RE.search(text)
                if sm:
                    for clause, point in point_refs(text):
                        suspension[(key[0], key[1], clause, point)] = compact(sm.group(0))

    facts = []
    for u in units:
        md = u.metadata or {}
        unit_type = md.get("unit_type")
        if unit_type not in {"point", "clause"}:
            continue
        doc_id = md.get("doc_id") or md.get("source") or ""
        article_no = str(md.get("article_number") or "")
        clause_no = str(md.get("clause_number") or "")
        point = str(md.get("point_letter") or "")
        article_title = compact(md.get("article") or "")
        clause_lead = compact(md.get("clause_lead") or "")
        point_text = compact(md.get("point_text") or "")
        body = u.page_content or ""
        fine_min, fine_max, fine_text = fine_from_text(clause_lead or body)
        if unit_type == "clause" and not fine_text:
            fine_min, fine_max, fine_text = fine_from_text(body)
        violation = point_text if unit_type == "point" and point_text else compact(body)
        if unit_type == "clause" and fine_text:
            # Remove the fine lead from broad clause cards so the violation field is not a duplicate of the fine.
            violation = compact(body.replace(fine_text, ""))[:500]
        vehicles = sorted(vehicle_profile(f"{article_title} {clause_lead} {point_text}"))
        fact_points = deduction.get((doc_id, article_no, clause_no, point))
        fact_suspension = suspension.get((doc_id, article_no, clause_no, point))
        fact_type = classify_fact_type(article_title, f"{clause_lead} {point_text} {body}", fine_text, fact_points, fact_suspension)
        answer_ready = bool(fine_text or fact_points or fact_suspension)
        fact = {
            "fact_id": str(md.get("unit_id") or ""),
            "source_unit_id": str(md.get("unit_id") or ""),
            "doc_id": doc_id,
            "article_number": article_no,
            "clause_number": clause_no,
            "point_letter": point,
            "unit_type": unit_type,
            "fact_type": fact_type,
            "answer_ready": answer_ready,
            "citation": " ".join(x for x in [doc_id, f"Điều {article_no}" if article_no else "", f"khoản {clause_no}" if clause_no else "", f"điểm {point}" if point else ""] if x),
            "article_title": article_title,
            "vehicle_scope": vehicles,
            "violation_text": violation[:700],
            "fine_min_vnd": fine_min,
            "fine_max_vnd": fine_max,
            "fine_text": fine_text,
            "points_deducted": fact_points,
            "suspension_text": fact_suspension,
            "source_path": md.get("source_path") or "",
        }
        if answer_ready or fact_type in {"sanction_context", "context"} and unit_type == "point":
            facts.append(fact)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
    coverage = {
        "facts": len(facts),
        "with_fine": sum(bool(f["fine_text"]) for f in facts),
        "with_vehicle": sum(bool(f["vehicle_scope"]) for f in facts),
        "with_points_deducted": sum(bool(f["points_deducted"]) for f in facts),
        "with_suspension": sum(bool(f["suspension_text"]) for f in facts),
        "answer_ready": sum(bool(f["answer_ready"]) for f in facts),
        "fact_type_counts": {t: sum(f.get("fact_type") == t for f in facts) for t in sorted({f.get("fact_type") for f in facts})},
        "output": str(out),
    }
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
