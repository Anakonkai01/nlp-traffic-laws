#!/usr/bin/env python
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class PairDS(Dataset):
    def __init__(self,path):
        self.rows=[json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate(tok,max_len):
    def fn(batch):
        enc=tok([b['question'] for b in batch],[b['text'] for b in batch],padding=True,truncation=True,max_length=max_len,return_tensors='pt')
        enc['labels']=torch.tensor([float(b['label']) for b in batch],dtype=torch.float)
        return enc
    return fn

@torch.no_grad()
def eval_model(model,loader,device):
    model.eval(); loss=0; n=0; correct=0
    for batch in loader:
        labels=batch.pop('labels').to(device); batch={k:v.to(device) for k,v in batch.items()}
        logits=model(**batch).logits.squeeze(-1); l=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
        loss+=l.item()*len(labels); n+=len(labels); correct+=(((torch.sigmoid(logits)>0.5).float()==labels).sum().item())
    return {'loss':loss/max(1,n),'acc':correct/max(1,n)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',default='data/training/fact_reranker_pairs_train.jsonl'); ap.add_argument('--dev',default='data/training/fact_reranker_pairs_dev.jsonl'); ap.add_argument('--model',default='BAAI/bge-reranker-v2-m3'); ap.add_argument('--output',default='models/fact-reranker-v1'); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--bs',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--max-length',type=int,default=384); ap.add_argument('--no-amp',action='store_true'); args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; tok=AutoTokenizer.from_pretrained(args.model); model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=1,ignore_mismatched_sizes=True).to(device)
    tr=DataLoader(PairDS(args.train),batch_size=args.bs,shuffle=True,collate_fn=collate(tok,args.max_length)); dv=DataLoader(PairDS(args.dev),batch_size=args.bs,shuffle=False,collate_fn=collate(tok,args.max_length))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr); steps=len(tr)*args.epochs; sched=get_linear_schedule_with_warmup(opt,max(1,steps//20),steps); scaler=torch.cuda.amp.GradScaler(enabled=(device=='cuda' and not args.no_amp))
    best=9e9; Path(args.output).mkdir(parents=True,exist_ok=True)
    for ep in range(args.epochs):
        model.train(); random.seed(42+ep)
        for batch in tr:
            labels=batch.pop('labels').to(device); batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda' and not args.no_amp)):
                logits=model(**batch).logits.squeeze(-1); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        met=eval_model(model,dv,device); print(json.dumps({'epoch':ep+1,**met}))
        if met['loss']<best:
            best=met['loss']; model.save_pretrained(args.output); tok.save_pretrained(args.output); Path(args.output,'ft_meta.json').write_text(json.dumps({'best_dev':met,'args':vars(args)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
