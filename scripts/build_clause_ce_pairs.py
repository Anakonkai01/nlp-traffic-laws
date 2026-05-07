#!/usr/bin/env python
"""Build CrossEncoder pairs for legal-unit reranking.

Fast version: chooses positives/negatives by token overlap inside the same
objective doc/article candidate pool instead of expensive ROUGE over all units.
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from legal_units import load_legal_units  # noqa: E402

ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)")
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
STOP = set("và của là có được trong với theo tại từ đến hoặc một các cho người điều khiển xe hành vi vi phạm".split())

def toks(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in STOP]

def overlap_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    common = sum((ca & cb).values())
    return common / max(1, min(len(a_tokens), len(b_tokens)))

def rows(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def article_no(row: dict, text: str) -> str:
    m = ARTICLE_RE.search(str(row.get("article") or "")) or ARTICLE_RE.search(text or "")
    return m.group(1) if m else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/splits/qa_train.jsonl")
    ap.add_argument("--output", default="data/training/clause_ce_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--negatives", type=int, default=6)
    ap.add_argument("--pos-threshold", type=float, default=0.22)
    ap.add_argument("--false-neg-threshold", type=float, default=0.16)
    ap.add_argument("--max-candidates", type=int, default=180)
    args = ap.parse_args()

    data = rows(Path(args.input))
    if args.limit:
        data = data[: args.limit]
    units = load_legal_units()
    unit_tok = {id(u): toks(u.page_content or "") for u in units}
    by_doc_article, by_doc = {}, {}
    for u in units:
        md = u.metadata or {}
        did = md.get("doc_id") or md.get("source") or ""
        art = str(md.get("article_number") or "")
        by_doc.setdefault(did, []).append(u)
        by_doc_article.setdefault((did, art), []).append(u)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with out.open("w", encoding="utf-8") as f:
        for r in data:
            q = r.get("question") or ""; gold = r.get("context") or r.get("answer") or ""
            if not q or not gold or str(r.get("corpus") or "") == "negative":
                skipped += 1; continue
            did = r.get("doc_id") or r.get("source") or ""; art = article_no(r, gold)
            cand = by_doc_article.get((did, art)) or by_doc.get(did) or []
            if not cand:
                skipped += 1; continue
            cand = cand[: args.max_candidates]
            gt = toks(gold)
            scored = [(u, overlap_score(gt, unit_tok[id(u)])) for u in cand]
            scored.sort(key=lambda x: -x[1])
            if scored[0][1] < args.pos_threshold:
                skipped += 1; continue
            pos, ps = scored[0]
            negs = [u.page_content for u, s in scored[1:] if s < args.false_neg_threshold][: args.negatives]
            if len(negs) < args.negatives:
                # Add same-document negatives from other articles to teach near-source mistakes.
                for u in by_doc.get(did, []):
                    if u.page_content != pos.page_content and u.page_content not in negs:
                        negs.append(u.page_content)
                    if len(negs) >= args.negatives: break
            if len(negs) < 2:
                skipped += 1; continue
            f.write(json.dumps({"question": q, "positive": pos.page_content, "negatives": negs, "positive_overlap": ps}, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "skipped": skipped, "output": str(out)}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
