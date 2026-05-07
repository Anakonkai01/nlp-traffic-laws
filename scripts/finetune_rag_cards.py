#!/usr/bin/env python
from __future__ import annotations
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import argparse, json, random
import unsloth
import torch
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, train_on_responses_only
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from config import MODEL_ID, TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT, TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT, KB_SOURCE_POLICY

def read(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('source_policy')!=KB_SOURCE_POLICY: continue
        rows.append(r)
    return rows

def fmt(r,tok,keep_prob):
    keep=random.random()<keep_prob
    if keep:
        user=f"Đoạn văn bản luật:\n{r['context']}\n\nCâu hỏi: {r['question']}"; sys_prompt=TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT
    else:
        user=f"Câu hỏi: {r['question']}"; sys_prompt=TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT
    return {'text':tok.apply_chat_template([{'role':'system','content':sys_prompt},{'role':'user','content':user},{'role':'assistant','content':r['answer']}],tokenize=False,add_generation_prompt=False)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',default='data/splits_rag_cards/qa_train.jsonl'); ap.add_argument('--dev',default='data/splits_rag_cards/qa_dev.jsonl'); ap.add_argument('--output',default='models/qwen3.5-9b-lora-traffic-rag-cards-v1'); ap.add_argument('--epochs',type=float,default=1); ap.add_argument('--bs',type=int,default=2); ap.add_argument('--ga',type=int,default=8); ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--max-seq-len',type=int,default=2048); ap.add_argument('--keep-context',type=float,default=0.9); args=ap.parse_args()
    model,tok=FastLanguageModel.from_pretrained(model_name=MODEL_ID,max_seq_length=args.max_seq_len,dtype=None,load_in_4bit=True)
    model=FastLanguageModel.get_peft_model(model,r=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_alpha=64,lora_dropout=0,bias='none',use_gradient_checkpointing='unsloth',random_state=42)
    tr=Dataset.from_list(read(args.train)).map(lambda x:fmt(x,tok,args.keep_context)); dv=Dataset.from_list(read(args.dev)).map(lambda x:fmt(x,tok,args.keep_context))
    trainer=SFTTrainer(model=model,tokenizer=tok,train_dataset=tr,eval_dataset=dv,max_seq_length=args.max_seq_len,args=SFTConfig(dataset_text_field='text',per_device_train_batch_size=args.bs,per_device_eval_batch_size=1,gradient_accumulation_steps=args.ga,num_train_epochs=args.epochs,learning_rate=args.lr,fp16=not torch.cuda.is_bf16_supported(),bf16=torch.cuda.is_bf16_supported(),optim='adamw_8bit',lr_scheduler_type='cosine',warmup_ratio=0.05,logging_steps=10,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,output_dir=str(Path(args.output)/'checkpoints'),report_to='none',seed=42))
    trainer=train_on_responses_only(trainer,instruction_part='<|im_start|>user\n',response_part='<|im_start|>assistant\n')
    trainer.train(); Path(args.output).mkdir(parents=True,exist_ok=True); model.save_pretrained(args.output); tok.save_pretrained(args.output); Path(args.output,'rag_card_meta.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8'); print('saved',args.output)
if __name__=='__main__': main()
