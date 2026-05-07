#!/usr/bin/env python
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from sanction_facts import load_sanction_facts, fact_to_card, fact_to_text
from scripts.build_fact_reranker_data import score_fact, articles

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--facts',default='data/legal_sanction_facts.jsonl'); ap.add_argument('--top-k',type=int,default=6); args=ap.parse_args()
 facts=load_sanction_facts(Path(args.facts)); by={}
 for f in facts: by.setdefault((str(f.get('doc_id') or ''),str(f.get('article_number') or '')),[]).append(f)
 out=[]; skipped=0
 for line in Path(args.input).read_text(encoding='utf-8').splitlines():
  if not line.strip(): continue
  r=json.loads(line); doc=str(r.get('doc_id') or r.get('source') or ''); arts=articles(f"{r.get('article','')} {r.get('context','')}")
  cand=[]
  for a in arts: cand.extend(by.get((doc,a),[]))
  if not cand: cand=[f for f in facts if str(f.get('doc_id') or '')==doc][:300]
  if not cand: skipped+=1; continue
  ranked=[f for s,f in sorted(((score_fact(r,f),f) for f in cand),key=lambda x:-x[0])[:args.top_k]]
  ctx='\n\n'.join(fact_to_card(f) for f in ranked)
  nr=dict(r); nr['context']=ctx; nr['corpus']='local_text'; nr['rag_fact_card_sft']=True; nr['fact_ids']=[f.get('fact_id') for f in ranked]
  out.append(nr)
 Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8')
 print(json.dumps({'input':args.input,'output':args.output,'rows':len(out),'skipped':skipped},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
