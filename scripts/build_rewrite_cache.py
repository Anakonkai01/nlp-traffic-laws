"""Precompute query rewrites for the 145-sample eval set.

Uses OpenRouter (gemini-2.0-flash-001) to rewrite colloquial traffic queries
into legal-document style. Cache is saved as JSON and read by src/evaluate.py
when RAG_QUERY_REWRITE=1.

The LoRA QA model can't do this (it always answers instead of rewriting),
so we delegate the rewrite to a small judge-style model. Cost: ~$0.005 per
145 queries.

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python scripts/build_rewrite_cache.py \
    --test-file data/eval_manual_labeled_v5.jsonl \
    --output data/query_rewrite_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

REWRITE_SYSTEM = (
    "Bạn là chuyên gia luật giao thông đường bộ Việt Nam. "
    "Hãy viết lại câu hỏi sau bằng văn phong pháp lý chuẩn của Nghị định 168/2024/NĐ-CP, "
    "Luật 35/2024/QH15 (Luật Đường bộ), và Luật 36/2024/QH15 (Luật Trật tự ATGT). "
    "Mục đích là để dùng câu viết lại làm truy vấn cho hệ thống tra cứu văn bản pháp luật.\n\n"
    "Quy tắc:\n"
    "1. GIỮ NGUYÊN ý nghĩa, đối tượng (xe gì), tình huống, và mọi con số/định lượng cụ thể trong câu gốc.\n"
    "2. Thay thuật ngữ khẩu ngữ bằng thuật ngữ pháp lý:\n"
    "   - 'vượt đèn đỏ' / 'vượt đèn vàng' -> 'không chấp hành hiệu lệnh của đèn tín hiệu giao thông'\n"
    "   - 'say rượu' / 'uống bia' / 'uống rượu' -> 'điều khiển xe trong khi trong máu hoặc hơi thở có nồng độ cồn'\n"
    "   - 'xe máy' (đứng một mình) -> 'xe mô tô, xe gắn máy'\n"
    "   - 'ô tô' (đứng một mình) -> 'xe ô tô'\n"
    "   - 'bằng lái' -> 'giấy phép lái xe'\n"
    "   - 'phạt nguội' -> 'xử phạt vi phạm hành chính thông qua thiết bị kỹ thuật nghiệp vụ'\n"
    "   - 'sai làn' -> 'điều khiển xe không đi đúng làn đường quy định'\n"
    "3. Nếu câu hỏi đã ở văn phong pháp lý thì giữ NGUYÊN VĂN.\n"
    "4. Không thêm thông tin mới (số điều, khoản, mức phạt) mà câu gốc không nêu.\n"
    "5. KHÔNG trả lời câu hỏi. KHÔNG thêm giải thích. KHÔNG thêm tiền tố/hậu tố.\n"
    "6. Output chỉ là MỘT câu hỏi viết lại trên một dòng."
)


def rewrite_one(client: OpenAI, model: str, question: str) -> str | None:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user",   "content": f"Câu hỏi gốc: {question}\n\nCâu hỏi pháp lý:"},
                ],
                max_tokens=160,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Strip leading labels / quotes
            text = re.sub(r"^[\"'`*\-•\s]+", "", text)
            text = re.sub(r"^Câu hỏi pháp lý[:\-]*\s*", "", text)
            text = text.strip().splitlines()[0] if text else ""
            if text and len(text) > 5:
                return text
            return None
        except Exception as exc:
            if attempt == 2:
                print(f"    error (final): {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/eval_manual_labeled_v5.jsonl")
    parser.add_argument("--output",    default="data/query_rewrite_cache.json")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    test_path = Path(args.test_file)
    samples = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(samples)} samples from {test_path}")

    cache_path = Path(args.output)
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded existing cache: {len(cache)} entries")

    new = 0
    for s in tqdm(samples, desc="rewrite"):
        q = s["question"]
        if q in cache:
            continue
        r = rewrite_one(client, args.model, q)
        if r:
            cache[q] = r
            new += 1
            if new % 10 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(cache)} entries to {cache_path}; {new} new")
    # Show a few rewrites
    for s in samples[:5]:
        q = s["question"]
        r = cache.get(q, "(missing)")
        print(f"  Q: {q[:90]}")
        print(f"  R: {r[:120]}")
        print()


if __name__ == "__main__":
    main()
