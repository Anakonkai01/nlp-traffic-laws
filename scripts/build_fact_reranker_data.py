#!/usr/bin/env python
"""Build structured fact reranker pairs from QA contexts and sanction facts.

Labels are slot/evidence based, not question-to-law rules. Positives are facts
whose fields overlap the gold answer/context; negatives are near misses from the
same retrieved candidate pool.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from sanction_facts import load_sanction_facts, fact_to_text  # noqa: E402

FINE_RE = re.compile(r"(?:\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|nghìn|triệu)", re.I)
ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.I)
VEHICLES = ["xe máy chuyên dùng", "ô tô", "xe máy", "mô tô", "xe gắn máy", "xe đạp", "người đi bộ"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").lower().strip()


def tok(s: str) -> set[str]:
    return {t for t in re.findall(r"[\wÀ-ỹ]+", norm(s)) if len(t) > 1}


def fines(s: str) -> set[str]:
    return {re.sub(r"\s+", " ", m.group(0)).lower() for m in FINE_RE.finditer(s or "")}


def articles(s: str) -> set[str]:
    return set(ARTICLE_RE.findall(s or ""))


def vehicles(s: str) -> set[str]:
    low = norm(s); return {v for v in VEHICLES if v in low}


def recall(gold: set[str], pred: set[str]) -> float:
    return len(gold & pred) / len(gold) if gold else 0.0


def score_fact(row: dict, fact: dict) -> float:
    gold = f"{row.get('answer','')} {row.get('context','')} {row.get('article','')}"
    ft = fact_to_text(fact)
    score = 0.0
    score += 3.0 * recall(fines(gold), fines(ft))
    score += 1.5 * recall(vehicles(gold), vehicles(ft))
    score += 1.0 * recall(articles(gold), {str(fact.get('article_number') or '')} | articles(ft))
    qtok = tok(row.get('question','')); vtok = tok(fact.get('violation_text',''))
    score += 1.0 * (len(qtok & vtok) / max(1, len(qtok)))
    if fact.get('answer_ready'): score += 0.25
    return score


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--qa', default='data/splits/qa_train.jsonl')
    ap.add_argument('--facts', default='data/legal_sanction_facts.jsonl')
    ap.add_argument('--output', default='data/training/fact_reranker_pairs.jsonl')
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--negatives', type=int, default=4)
    args=ap.parse_args()
    rows=[json.loads(l) for l in Path(args.qa).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.max_rows: rows=rows[:args.max_rows]
    facts=load_sanction_facts(Path(args.facts))
    by_doc_article={}
    for f in facts:
        by_doc_article.setdefault((str(f.get('doc_id') or ''), str(f.get('article_number') or '')), []).append(f)
    out=[]; pos=neg=skipped=0
    for row in rows:
        doc=str(row.get('doc_id') or row.get('source') or '')
        arts=articles(f"{row.get('article','')} {row.get('context','')}")
        cand=[]
        if arts:
            for a in arts: cand.extend(by_doc_article.get((doc,a), []))
        if not cand and doc:
            cand=[f for f in facts if str(f.get('doc_id') or '')==doc][:300]
        if not cand:
            skipped+=1; continue
        scored=sorted(((score_fact(row,f),f) for f in cand), key=lambda x:-x[0])
        positives=[(s,f) for s,f in scored if s>=2.5][:2]
        if not positives:
            skipped+=1; continue
        negative_pool=[(s,f) for s,f in scored if s<2.0]
        for s,f in positives:
            out.append({'question':row['question'], 'fact_id':f.get('fact_id'), 'text':fact_to_text(f), 'label':1.0, 'score':s})
            pos+=1
            for ns,nf in negative_pool[:args.negatives]:
                out.append({'question':row['question'], 'fact_id':nf.get('fact_id'), 'text':fact_to_text(nf), 'label':0.0, 'score':ns})
                neg+=1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out), encoding='utf-8')
    print(json.dumps({'rows':len(rows),'pairs':len(out),'positive':pos,'negative':neg,'skipped':skipped,'output':args.output}, ensure_ascii=False, indent=2))

if __name__=='__main__': main()
