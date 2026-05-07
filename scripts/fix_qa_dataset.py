"""
Post-process qa_pairs_traffic.jsonl to fix over-refusal issues:
1. Reduce negatives from 44.6% → ~10% (keep ~200 of 1286)
2. Add hard-context examples (~500): positive QAs with wrong context + "(Kiến thức chung)" marker
3. Merge in new penalty QAs from QA generation
4. Output cleaned dataset to a new file
"""
import json
import random
from pathlib import Path

random.seed(42)

REFUSAL_ANSWERS = [
    "Tôi không tìm thấy căn cứ trong các văn bản giao thông đã được cung cấp để trả lời câu hỏi này.",
    "Câu hỏi này không thuộc phạm vi pháp luật giao thông đường bộ hiện hành.",
    "Tôi không tìm thấy quy định cụ thể về vấn đề này trong các văn bản luật giao thông hiện hành.",
    "Văn bản được cung cấp không đề cập đến nội dung này. Tôi không thể trả lời dựa trên thông tin hiện có.",
    "Xin lỗi, tôi chưa có đủ thông tin để trả lời chính xác câu hỏi này.",
    "Nội dung này không nằm trong phạm vi điều chỉnh của các văn bản luật giao thông đã được cung cấp.",
]

DATA_PATH = Path(__file__).parent.parent / "data" / "qa_pairs_traffic_v2.jsonl"
PENALTY_PATH = Path(__file__).parent.parent / "data" / "qa_penalty.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "qa_pairs_traffic_v3.jsonl"
TEXT_DIR = Path(__file__).parent.parent / "docs" / "docs_giaothong" / "text"

# ── Load existing v2 data ──
with open(DATA_PATH, "r", encoding="utf-8") as f:
    v2_samples = [json.loads(line) for line in f if line.strip()]

positives = [s for s in v2_samples if s.get("corpus") == "local_text"]
negatives = [s for s in v2_samples if s.get("corpus") == "negative"]
hard_context = [s for s in v2_samples if s.get("corpus") == "hard_context"]

print(f"V2: {len(positives)} pos, {len(negatives)} neg, {len(hard_context)} hard")

# ── Merge new penalty QAs ──
new_penalties = []
if PENALTY_PATH.exists():
    with open(PENALTY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("corpus") == "local_text":
                    q = item["question"].strip().lower()
                    if not any(p["question"].strip().lower() == q for p in positives):
                        new_penalties.append(item)

print(f"New penalty QAs: {len(new_penalties)}")
positives.extend(new_penalties)

# ── Keep only 200 negatives with diverse answers ──
random.shuffle(negatives)
kept_negatives = negatives[:200]
for neg in kept_negatives:
    neg["answer"] = random.choice(REFUSAL_ANSWERS)

print(f"Negatives kept: {len(kept_negatives)}")

# ── Create 500 hard-context examples ──
source_chunks = {}
for filename in ["nd_168_2024_nd_cp.txt", "nd_158_2024_nd_cp.txt", "nd_165_2024_nd_cp.txt",
                   "nd_336_2025_nd_cp.txt", "tt_65_2024_tt_bca.txt", "luat_35_2024_qh15.txt",
                   "luat_36_2024_qh15.txt", "qcvn_41_2024_bgtvt.txt", "tt_05_2025_tt_bgtvt.txt",
                   "nd_39_2023_nd_cp.txt", "tt_79_2024_tt_bca.txt"]:
    p = TEXT_DIR / filename
    if p.exists():
        source_chunks[filename.replace(".txt", "")] = p.read_text(encoding="utf-8")[:5000]

rng = random.Random(42)
rng.shuffle(positives)

target_hard = min(500, len(positives) // 3)
hard_created = []
for p in positives:
    if len(hard_created) >= target_hard:
        break
    doc_id = p.get("doc_id", "")
    wrong_docs = [d for d in source_chunks if d != doc_id]
    if not wrong_docs:
        continue
    wrong_doc = rng.choice(wrong_docs)
    wrong_chunk = source_chunks[wrong_doc][:2000]
    answer = p["answer"]
    if "(Kiến thức chung)" not in answer:
        answer = answer.rstrip(".") + " (Kiến thức chung)"
    hard_created.append({
        "question": p["question"],
        "answer": answer,
        "context": wrong_chunk,
        "article": "",
        "source": wrong_doc,
        "doc_id": wrong_doc,
        "title": "",
        "authority": "",
        "source_path": "",
        "corpus": "hard_context",
        "source_policy": "local_text_only",
    })

print(f"Hard-context created: {len(hard_created)}")

# ── Write output ──
output = positives + kept_negatives + hard_created
rng.shuffle(output)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

pos_final = sum(1 for s in output if s.get("corpus") == "local_text")
neg_final = sum(1 for s in output if s.get("corpus") == "negative")
hard_final = sum(1 for s in output if s.get("corpus") == "hard_context")
total = len(output)
print(f"\n{'='*50}")
print(f"Cleaned dataset → {OUTPUT_PATH}")
print(f"  Total: {total}")
print(f"  Positive: {pos_final} ({pos_final/total*100:.1f}%)")
print(f"  Negative: {neg_final} ({neg_final/total*100:.1f}%)")
print(f"  Hard-context: {hard_final} ({hard_final/total*100:.1f}%)")
print(f"{'='*50}")
