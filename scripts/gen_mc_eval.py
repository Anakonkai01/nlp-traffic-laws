"""
Generate multiple-choice (MC) evaluation candidates from local traffic law documents.

Usage:
  cd nlp
  python scripts/gen_mc_eval.py --count 20 --output data/mc_candidates.jsonl
  # Review mc_candidates.jsonl, then append good ones to data/eval_mc_manual.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking import article_clause_chunks
from config import OLLAMA_MODEL
from corpus import iter_traffic_corpus_records, infer_article
from langchain_core.documents import Document

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật và sinh 1 câu hỏi trắc nghiệm 4 lựa chọn.
Yêu cầu:
- Câu hỏi và đáp án đúng phải bám sát đoạn văn bản, không suy diễn
- 3 đáp án sai phải hợp lý (không quá dễ loại trừ), nhưng rõ ràng là sai theo đoạn văn
- Choices KHÔNG có tiền tố A/B/C/D — chỉ viết nội dung đáp án
- Chỉ trả về JSON, không giải thích thêm
Format bắt buộc (choices là mảng 4 chuỗi, không có A./B./C./D. ở đầu):
{"question": "...", "choices": ["nội dung đáp án 1", "nội dung đáp án 2", "nội dung đáp án 3", "nội dung đáp án 4"], "answer": "A"}"""


def _call_ollama(chunk: str, model: str, timeout: int) -> dict | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Đoạn văn bản luật:\n{chunk}\n\nSinh 1 câu hỏi trắc nghiệm.",
                    },
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": 400, "temperature": 0.3, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Extract JSON
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.startswith("json"):
                content = content[4:].strip()
        import json as _json
        parsed = _json.loads(content)
        return parsed
    except Exception:
        return None


def _validate(item: dict, chunk: str) -> bool:
    if not isinstance(item, dict):
        return False
    if not all(k in item for k in ("question", "choices", "answer")):
        return False
    if not isinstance(item["choices"], list) or len(item["choices"]) != 4:
        return False
    if item["answer"] not in ("A", "B", "C", "D"):
        return False
    if len(item["question"]) < 10:
        return False
    return True


def generate_mc_candidates(
    count: int,
    output: Path,
    model: str = OLLAMA_MODEL,
    timeout: int = 60,
    min_chunk: int = 100,
    seed: int = 42,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    collected = 0
    attempts = 0

    with output.open("w", encoding="utf-8") as f:
        for pass_idx in range(10):
            if collected >= count:
                break
            for record in iter_traffic_corpus_records(shuffle_seed=seed + pass_idx):
                if collected >= count:
                    break
                doc = Document(
                    page_content=record["text"],
                    metadata={"doc_id": record["doc_id"]},
                )
                chunks = [
                    c.page_content
                    for c in article_clause_chunks(doc)
                    if len(c.page_content) >= min_chunk
                ]
                if not chunks:
                    continue
                chunk = rng.choice(chunks)
                attempts += 1
                start = time.time()
                result = _call_ollama(chunk, model, timeout)
                elapsed = time.time() - start

                if result and _validate(result, chunk):
                    item = {
                        "question": result["question"],
                        "choices": result["choices"],
                        "answer": result["answer"],
                        "context": chunk,
                        "article": infer_article(chunk),
                        "source": record["doc_id"],
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    collected += 1
                    print(f"[{collected}/{count}] {elapsed:.1f}s | {item['question'][:60]}")
                else:
                    print(f"  skip (attempt {attempts}, {elapsed:.1f}s)")

    print(f"\nDone: {collected} MC candidates → {output}")
    print(f"Review and append to data/eval_mc_manual.jsonl:")
    print(f"  cat {output} >> data/eval_mc_manual.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/mc_candidates.jsonl"))
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_mc_candidates(
        count=args.count,
        output=args.output,
        model=args.model,
        timeout=args.timeout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
