#!/usr/bin/env python
"""Build RAG-card SFT data from train/dev splits using oracle/gold context cards.

This trains the generator to read compact evidence cards instead of raw chunks.
It is data-driven: cards are derived from each sample's gold context/source, not
from eval_manual or question-specific rules.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from evidence_cards import evidence_card_from_text  # noqa:E402


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--max-rows',type=int,default=0); args=ap.parse_args()
    out=[]
    for i,line in enumerate(Path(args.input).read_text(encoding='utf-8').splitlines()):
        if args.max_rows and len(out)>=args.max_rows: break
        if not line.strip(): continue
        r=json.loads(line); ctx=r.get('context') or ''
        if not ctx: continue
        md={k:r.get(k,'') for k in ['doc_id','source','article','title','source_path','corpus']}
        card=evidence_card_from_text(r.get('question',''), ctx, md)
        nr=dict(r); nr['context']=card; nr['corpus']='local_text'; nr['rag_card_sft']=True
        out.append(nr)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8')
    print(json.dumps({'input':args.input,'output':args.output,'rows':len(out)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
