## NLP Traffic-Law Pipeline

Pipeline `nlp/` hiện dùng chính sách **local text only**. KB, sinh QA, fine-tune và eval chỉ được đọc các file `.txt` đã bật trong:

```bash
nlp/docs/docs_giaothong/manifest.json
```

Không dùng corpus VLSP, không fallback sang PDF, không tự crawl dữ liệu khi chạy production path.

### Nguồn Dữ Liệu

- Đặt văn bản luật dạng UTF-8 `.txt` trong `nlp/docs/docs_giaothong/text/`.
- Thêm hoặc bật văn bản trong `nlp/docs/docs_giaothong/manifest.json`.
- Chỉ các entry `enabled: true` và `source_policy: local_text_only` mới được đưa vào pipeline.
- Nếu cần thêm Luật 35/2024/QH15 hoặc Luật 36/2024/QH15, hãy lưu bản text vào thư mục `text/` rồi thêm vào manifest.

### Artifact

- QA dataset: `nlp/data/qa_pairs_traffic.jsonl`
- Manual eval: `nlp/data/eval_manual.jsonl`
- Manual MC eval: `nlp/data/eval_mc_manual.jsonl`
- Vector DB: `nlp/vector_db_traffic/`
- KB metadata guardrail: `nlp/vector_db_traffic/build_meta.json`
- LoRA adapter: `nlp/models/qwen3.5-9b-lora-traffic/`
- Reports: `nlp/reports/traffic/`

`load_vectorstore()` sẽ từ chối KB thiếu `build_meta.json` hoặc không có `source_policy=local_text_only`, để tránh demo load nhầm artifact cũ.

### Workflow

```bash
cd nlp
python scripts/filter_traffic_laws.py
python src/build_kb.py --force
python src/generate_qa.py --force
python scripts/filter_qa.py
python src/finetune.py
python src/evaluate.py --configs A B C D
python src/evaluate_mc.py --configs A B C D
python src/app.py
```

Hoặc chạy pipeline tuần tự:

```bash
REBUILD_KB=1 REGENERATE_QA=1 bash nlp/auto_pipeline.sh
```

QA generation nhanh để smoke:

```bash
cd nlp
python src/generate_qa.py --force --profile smoke
python src/generate_qa.py --force --phases penalties procedures --phase-size 20 --no-negatives --timeout 180 --num-predict 160
python src/generate_qa.py --phases penalties --phase-size 50 --max-workers 1 --max-passes 4 --timeout 180 --num-predict 160
```

### Production Notes

- Demo mặc định ưu tiên Config D: fine-tuned model + local-text-only RAG.
- Prompt yêu cầu chỉ trả lời dựa trên context truy xuất và từ chối khi thiếu căn cứ.
- RAG context có metadata nguồn: `doc_id`, tiêu đề, file text, điều khoản nếu tách được.
- Eval báo thêm `forbidden_legacy_rate` để bắt câu trả lời viện dẫn nguồn cũ như Luật Giao thông đường bộ 2008, NĐ 100/2019, NĐ 46/2016 hoặc dự thảo.
