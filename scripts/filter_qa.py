"""
Filter và thống kê QA pairs cho fine-tune.

Loại bỏ:
  1. Answers > 500 ký tự (tránh LM học verbosity)
  2. Questions bắt đầu bằng "Điều" (template artifact)

Output:
  - data/qa_pairs_traffic.jsonl              ghi đè (file sạch)
  - reports/traffic/qa_filter_report.json     báo cáo chi tiết cho final report
"""
import json, re
from pathlib import Path

QA_PATH = Path("data/qa_pairs_traffic.jsonl")
REPORT_DIR = Path("reports/traffic")
ANSWER_MAX_CHARS = 500

pairs = []
with open(QA_PATH) as f:
    for l in f:
        if l.strip():
            pairs.append(json.loads(l))

total = len(pairs)
pos = [p for p in pairs if p.get("corpus") == "local_text"]
neg = [p for p in pairs if p.get("corpus") == "negative"]

# ── Filters ────────────────────────────────────────────────────────────────────

# Filter 1: Long answers
long_answer_ids = set()
for i, p in enumerate(pairs):
    if p.get("corpus") == "local_text" and len(p.get("answer", "")) > ANSWER_MAX_CHARS:
        long_answer_ids.add(i)

# Filter 2: Điều-template questions
dieu_template_ids = set()
for i, p in enumerate(pairs):
    if p.get("corpus") == "local_text":
        q = p.get("question", "").strip()
        if q.startswith("Điều") and ("văn bản này" in q or "quy định về vấn đề" in q):
            dieu_template_ids.add(i)

all_bad = long_answer_ids | dieu_template_ids
filtered = [p for i, p in enumerate(pairs) if i not in all_bad]

# ── Stats ──────────────────────────────────────────────────────────────────────

pos_after = len([p for p in filtered if p.get("corpus") == "local_text"])
neg_after = len([p for p in filtered if p.get("corpus") == "negative"])

report = {
    "before": {
        "total": total,
        "positive": len(pos),
        "negative": len(neg),
    },
    "filters_applied": {
        "long_answers_max_chars": ANSWER_MAX_CHARS,
        "long_answers_removed": len(long_answer_ids),
        "dieu_template_removed": len(dieu_template_ids),
        "total_removed": len(all_bad),
    },
    "after": {
        "total": len(filtered),
        "positive": pos_after,
        "negative": neg_after,
    },
    "retention": {
        "total_pct": round(len(filtered) / total * 100, 1),
        "positive_pct": round(pos_after / len(pos) * 100, 1) if pos else 0,
    },
}

# ── Save ───────────────────────────────────────────────────────────────────────

REPORT_DIR.mkdir(parents=True, exist_ok=True)
with open(QA_PATH, "w", encoding="utf-8") as f:
    for p in filtered:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

report_path = REPORT_DIR / "qa_filter_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ── Print ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("QA FILTER REPORT — có thể dùng trong báo cáo cuối kỳ")
print("=" * 60)
print(f"\n📊 Before:")
print(f"   Total:     {total}")
print(f"   Positive:  {len(pos)}")
print(f"   Negative:  {len(neg)}")
print(f"\n🔍 Filters:")
print(f"   Answers > {ANSWER_MAX_CHARS} chars:  {len(long_answer_ids)} removed")
print(f"   'Điều X...' templates:     {len(dieu_template_ids)} removed")
print(f"   Total:                     {len(all_bad)} removed")
print(f"\n📊 After:")
print(f"   Total:     {len(filtered)}")
print(f"   Positive:  {pos_after}")
print(f"   Negative:  {neg_after}")
print(f"\n📈 Retention: {report['retention']['total_pct']}%")
print(f"\n📁 Saved filtered → {QA_PATH}")
print(f"📁 Saved report  → {report_path}")
