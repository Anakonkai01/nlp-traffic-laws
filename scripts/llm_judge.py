"""LLM-as-Judge via OpenRouter.

Scores predictions for configs A/B/C/D on a 1-5 scale, normalised to [0,1].
Reads A/B/C from reports/traffic/predictions_all_configs.json and
D from reports/traffic/preds_d_clean_bertscore.json (final clean run).
Writes llm_judge per config into reports/traffic/evaluation_results.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cheap pick: gemini-2.0-flash ~$0.10/$0.40 per M token, ~$0.05 for 4x145 calls
MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.0-flash-001")

JUDGE_SYSTEM = (
    "Bạn là chuyên gia đánh giá hệ thống hỏi đáp về Luật Giao thông Đường bộ Việt Nam. "
    "Nhiệm vụ của bạn là chấm điểm câu trả lời của mô hình AI so với đáp án chuẩn. "
    "Chỉ trả lời bằng đúng một chữ số từ 1 đến 5, không giải thích thêm."
)

JUDGE_TEMPLATE = """\
Câu hỏi: {question}

Đáp án chuẩn: {reference}

Câu trả lời của mô hình: {prediction}

Thang điểm:
5 — Hoàn toàn chính xác và đầy đủ
4 — Phần lớn chính xác, thiếu vài chi tiết nhỏ
3 — Đúng về đại thể nhưng thiếu thông tin quan trọng
2 — Có phần đúng nhưng sai nhiều điểm
1 — Sai hoàn toàn hoặc không liên quan

Điểm (1-5):"""


def judge_one(client: OpenAI, question: str, reference: str, prediction: str) -> int | None:
    prompt = JUDGE_TEMPLATE.format(question=question, reference=reference, prediction=prediction)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"[1-5]", text)
            if m:
                return int(m.group())
            return None
        except Exception as exc:
            if attempt == 2:
                print(f"    judge error (final): {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def judge_config(client, cfg: str, questions, references, predictions) -> float:
    scores: list[int] = []
    failed = 0
    for q, ref, pred in tqdm(zip(questions, references, predictions), total=len(predictions), desc=f"Judge {cfg}"):
        s = judge_one(client, q, ref, pred)
        if s is None:
            failed += 1
        else:
            scores.append(s)
    if not scores:
        print(f"  {cfg}: all failed")
        return 0.0
    mean = sum(scores) / len(scores)
    print(f"  {cfg}: mean {mean:.3f}/5 = {mean/5:.4f} (failed {failed}/{len(predictions)})")
    return mean / 5.0


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    root = Path(__file__).resolve().parent.parent
    reports = root / "reports" / "traffic"

    abc_preds = json.loads((reports / "predictions_all_configs.json").read_text(encoding="utf-8"))
    d_preds = json.loads((reports / "preds_d_clause_final.json").read_text(encoding="utf-8"))

    # preds_d_clean_bertscore only has config D. Override D with the final clean run.
    combined = {**abc_preds, "D": d_preds["D"]}

    results_path = reports / "evaluation_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    for cfg in ["A", "B", "C", "D"]:
        if cfg not in combined:
            continue
        block = combined[cfg]
        qs = block["questions"]
        refs = block["references"]
        preds = block["predictions"]
        judge = judge_config(client, cfg, qs, refs, preds)
        results.setdefault(cfg, {})["llm_judge"] = round(judge, 4)
        results[cfg]["llm_judge_model"] = MODEL
        # Save after each config so we don't lose progress on API hiccup.
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFinal scores (normalised 0-1):")
    for cfg in ["A", "B", "C", "D"]:
        if cfg in results and "llm_judge" in results[cfg]:
            print(f"  {cfg}: {results[cfg]['llm_judge']:.4f}")


if __name__ == "__main__":
    main()
