import argparse
import hashlib
import json
import os
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import XMLParsedAsHTMLWarning
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
load_dotenv(Path(__file__).parent.parent / ".env")

from chunking import article_clause_chunks
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FORBIDDEN_LEGACY_TERMS,
    KB_SOURCE_POLICY,
    OLLAMA_MODEL,
    QA_DATA_PATH,
    REFUSAL_ANSWER,
    SEPARATORS,
)
from corpus import infer_article, iter_traffic_corpus_records


MODEL = os.environ.get("OLLAMA_MODEL_OVERRIDE", OLLAMA_MODEL)
OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OUTPUT_PATH = QA_DATA_PATH
_CUSTOM_OUTPUT_PATH: Path | None = None
DEFAULT_MAX_WORKERS = 3
DEFAULT_MAX_PHASE_PASSES = 30
DEFAULT_OLLAMA_TIMEOUT = 45
DEFAULT_NUM_PREDICT = 450
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 2048
DEFAULT_NUM_GPU = None

LOCAL_GENERAL_TARGET = 250
LOCAL_DEFINITION_TARGET = 650
LOCAL_PENALTY_TARGET = 1400
LOCAL_PROCEDURE_TARGET = 1900
LOCAL_PROHIBITED_TARGET = 2400
LOCAL_SCENARIO_TARGET = 2800
NEG_FINAL_TARGET = 3000
NEG_SEED = 60
FORBIDDEN_LEGACY_TERMS_LOWER = [term.lower() for term in FORBIDDEN_LEGACY_TERMS]


SYSTEM_PROMPT_GENERAL = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời bằng tiếng Việt.
Yêu cầu:
- Câu hỏi phải có thể trả lời được từ đoạn văn bản
- Câu trả lời phải bám sát và ưu tiên trích gần nguyên văn từ đoạn văn bản
- Không suy diễn, không thêm nội dung không có trong đoạn văn, không thay đổi chủ thể
- Ưu tiên nêu văn bản, điều/khoản/điểm nếu đoạn văn có thông tin đó
- Chỉ trả về JSON, không giải thích thêm
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_DEFINITION = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời về ĐỊNH NGHĨA/KHÁI NIỆM bằng tiếng Việt.
Câu trả lời phải trích gần nguyên văn định nghĩa hoặc nội dung khái niệm trong đoạn văn.
Không suy diễn, không thêm nội dung không có trong đoạn văn, không thay đổi chủ thể.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_PENALTY = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời về XỬ PHẠT VI PHẠM bằng tiếng Việt.
Câu trả lời phải nêu đủ mức phạt tiền, hình thức xử phạt bổ sung, mức trừ điểm/tước GPLX nếu đoạn văn có quy định.
Câu trả lời phải bám sát và trích gần nguyên văn đoạn văn, không suy diễn, không thêm số liệu hay hành vi không có trong đoạn văn.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_PROCEDURE = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời về ĐIỀU KIỆN/THỦ TỤC bằng tiếng Việt.
Câu trả lời phải liệt kê đầy đủ bước, hồ sơ, điều kiện, thời hạn hoặc thẩm quyền nếu đoạn văn có quy định.
Câu trả lời phải bám sát nguyên văn đoạn văn, không suy diễn, không thêm nội dung hay điều kiện không có trong đoạn văn.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_PROHIBITED = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời về HÀNH VI BỊ CẤM/NGHĨA VỤ PHÁP LÝ bằng tiếng Việt.
Câu trả lời phải bám sát và trích gần nguyên văn đoạn văn, không suy diễn ngoài văn bản, không thay đổi chủ thể.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_SCENARIO = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời dạng TÌNH HUỐNG THỰC TẾ bằng tiếng Việt.
Câu hỏi phải mô tả tình huống cụ thể; câu trả lời phải nêu cách xử lý hoặc mức xử phạt đúng theo đoạn văn.
Câu trả lời phải bám sát nguyên văn đoạn văn, không suy diễn hay thêm hình phạt/điều kiện không có trong đoạn văn.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

SYSTEM_PROMPT_REALISTIC = """Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam.
Nhiệm vụ: đọc đoạn văn bản luật giao thông và sinh 1 cặp câu hỏi - câu trả lời dạng CÂU HỎI THỰC TẾ bằng tiếng Việt.
Viết câu hỏi như người dân bình thường hỏi về luật giao thông (ví dụ: "Em đi xe máy vượt đèn đỏ bị phạt bao nhiêu?", "Cho em hỏi thủ tục đăng ký xe mới mua").
Câu trả lời phải trích gần nguyên văn từ đoạn văn bản luật, chính xác, ngắn gọn, dễ hiểu.
Không suy diễn, không thêm nội dung không có trong đoạn văn, không thay đổi chủ thể.
Chỉ trả về JSON, không giải thích thêm.
Format bắt buộc: {"question": "...", "answer": "..."}"""

PHASE_CONFIGS = [
    ("general", "General", SYSTEM_PROMPT_GENERAL, 0, 80, LOCAL_GENERAL_TARGET),
    ("definitions", "Definitions", SYSTEM_PROMPT_DEFINITION, 10, 80, LOCAL_DEFINITION_TARGET),
    ("penalties", "Penalties", SYSTEM_PROMPT_PENALTY, 20, 80, LOCAL_PENALTY_TARGET),
    ("procedures", "Procedures", SYSTEM_PROMPT_PROCEDURE, 30, 100, LOCAL_PROCEDURE_TARGET),
    ("prohibited", "Prohibited", SYSTEM_PROMPT_PROHIBITED, 40, 80, LOCAL_PROHIBITED_TARGET),
    ("scenarios", "Scenarios", SYSTEM_PROMPT_SCENARIO, 50, 120, LOCAL_SCENARIO_TARGET),
    ("realistic", "Realistic", SYSTEM_PROMPT_REALISTIC, 60, 80, 4000),
]

SMOKE_PHASE_TARGETS = {
    "general": 24,
    "definitions": 48,
    "penalties": 96,
    "procedures": 128,
    "prohibited": 160,
    "scenarios": 192,
}
SMOKE_NEGATIVE_TARGET = 220

_TRIVIAL_Q_PATTERNS = [
    "có hiệu lực thi hành từ",
    "có hiệu lực từ ngày",
    "ban hành kèm theo",
    "ký ngày",
    "số hiệu văn bản",
    "ngày ban hành",
]

_CANT_ANSWER_PATTERNS = [
    "không nêu cụ thể",
    "không đề cập",
    "không cung cấp",
    "không có thông tin",
    "không đủ thông tin",
    "không tìm thấy",
]


def _get_output_path() -> Path:
    return _CUSTOM_OUTPUT_PATH if _CUSTOM_OUTPUT_PATH is not None else OUTPUT_PATH


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _answer_grounded(answer: str, context: str, min_recall: float = 0.35) -> bool:
    """Reject answers whose tokens don't overlap enough with the context (catches hallucination)."""
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    c_tokens = set(re.findall(r"\w+", context.lower()))
    if not a_tokens:
        return False
    return len(a_tokens & c_tokens) / len(a_tokens) >= min_recall


def _contains_forbidden_legacy(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in FORBIDDEN_LEGACY_TERMS_LOWER)


def _is_trivial_qa(question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()
    return any(pat in q for pat in _TRIVIAL_Q_PATTERNS) or any(
        pat in a for pat in _CANT_ANSWER_PATTERNS
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def call_ollama(
    chunk: str,
    system_prompt: str,
    model: str,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    debug: bool = False,
) -> dict | None:
    options = {
        "num_predict": num_predict,
        "temperature": temperature,
        "num_ctx": num_ctx,
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Đoạn văn bản luật:\n"
                            f"{chunk}\n\n"
                            "Sinh 1 cặp câu hỏi - câu trả lời dựa trên đoạn văn trên."
                        ),
                    },
                ],
                "stream": False,
                "think": False,
                "options": options,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        parsed = _extract_json(content)
        if debug:
            print(f"raw_response={content!r}")
            print(f"parsed={parsed!r}")
        return parsed
    except (requests.RequestException, KeyError, TypeError) as exc:
        if debug:
            print(f"ollama_error={type(exc).__name__}: {exc}")
        return None


def call_openrouter(
    chunk: str,
    system_prompt: str,
    model: str = OPENROUTER_MODEL,
    timeout: int = 120,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
    debug: bool = False,
) -> dict | None:
    """Generate QA via OpenRouter API (OpenAI-compatible)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai  # needed for OpenRouter backend")
        return None

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Đoạn văn bản luật:\n"
                        f"{chunk}\n\n"
                        "Sinh 1 cặp câu hỏi - câu trả lời dựa trên đoạn văn trên."
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=num_predict,
            seed=42,
            extra_body={
                "provider": {
                    "allow_fallbacks": True,
                    "order": [
                        "together", "venice", "google-ai-studio",
                        "nvidia", "novita", "liquid", "deepseek",
                    ],
                }
            },
        )
        content = resp.choices[0].message.content.strip()
        parsed = _extract_json(content)
        if debug:
            print(f"raw_response={content!r}")
            print(f"parsed={parsed!r}")
        return parsed
    except Exception as exc:
        if debug:
            print(f"openrouter_error={type(exc).__name__}: {exc}")
        return None


def _process(
    record: dict,
    system_prompt: str,
    min_chunk: int,
    seed: int,
    model: str,
    backend: str = "ollama",
    openrouter_model: str = OPENROUTER_MODEL,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_gpu: int | None = DEFAULT_NUM_GPU,
) -> dict | None:
    text = record.get("text") or ""
    if not text:
        return None

    doc = Document(
        page_content=text,
        metadata={
            "doc_id": record.get("doc_id", ""),
            "source": record.get("source", ""),
            "title": record.get("title", ""),
        },
    )
    raw_chunks = article_clause_chunks(doc)
    valid_chunks = [
        c.page_content for c in raw_chunks
        if len(c.page_content) >= min_chunk and not _contains_forbidden_legacy(c.page_content)
    ]
    if not valid_chunks:
        return None

    rng = random.Random(seed)
    chunk = rng.choice(valid_chunks)
    if backend == "openrouter":
        result = call_openrouter(
            chunk,
            system_prompt,
            model=openrouter_model,
            timeout=timeout,
            num_predict=num_predict,
        )
    else:
        result = call_ollama(
            chunk,
            system_prompt,
            model,
            timeout=timeout,
            num_predict=num_predict,
            num_ctx=num_ctx,
            num_gpu=num_gpu,
        )
    if not result or "question" not in result or "answer" not in result:
        return None
    question = str(result["question"]).strip()
    answer = str(result["answer"]).strip()
    if not question or not answer or _is_trivial_qa(question, answer):
        return None
    if not _answer_grounded(answer, chunk):
        return None

    return {
        "question": question,
        "answer": answer,
        "context": chunk,
        "article": infer_article(chunk),
        "source": record.get("source", ""),
        "doc_id": record.get("doc_id", ""),
        "title": record.get("title", ""),
        "authority": record.get("authority", ""),
        "source_path": record.get("source_path", ""),
        "corpus": "local_text",
        "source_policy": KB_SOURCE_POLICY,
    }


def _read_existing_questions() -> tuple[int, set[str]]:
    count = 0
    seen_questions: set[str] = set()
    out = _get_output_path()
    if not out.exists():
        return count, seen_questions

    with out.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("source_policy") != KB_SOURCE_POLICY and item.get("corpus") != "negative":
                raise ValueError(
                    f"{out} contains non-local sample at line {line_no}. "
                    "Run generate_qa.py --force to discard old generated data."
                )
            if item.get("corpus") not in {"local_text", "negative"}:
                raise ValueError(
                    f"{out} contains unsupported corpus={item.get('corpus')!r} "
                    f"at line {line_no}. Run generate_qa.py --force."
                )
            count += 1
            q = item.get("question")
            if q:
                seen_questions.add(q.strip().lower())
    return count, seen_questions


def _run_phase(
    phase_key: str,
    phase_name: str,
    system_prompt: str,
    seed: int,
    min_chunk: int,
    target: int,
    seen_questions: set[str],
    max_workers: int,
    max_passes: int,
    model: str,
    backend: str,
    openrouter_model: str,
    timeout: int,
    num_predict: int,
    num_ctx: int,
    num_gpu: int | None,
) -> None:
    count, _ = _read_existing_questions()
    if count >= target:
        print(f"{phase_name}: already at {count}/{target}, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"{phase_name}: generating pairs {count + 1}-{target}")
    print(
        f"  source_policy={KB_SOURCE_POLICY} seed={seed} min_chunk={min_chunk} "
        f"model={'openrouter:' + openrouter_model if backend == 'openrouter' else model} "
        f"backend={backend} timeout={timeout}s num_predict={num_predict} "
        f"num_ctx={num_ctx} num_gpu={num_gpu}"
    )
    print(f"{'=' * 60}")

    attempts = 0
    failures = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool, _get_output_path().open(
        "a", encoding="utf-8"
    ) as f:
        pbar = tqdm(total=target, initial=count, desc=phase_name, unit="pair")
        pending = []

        def _collect(future) -> None:
            nonlocal count, attempts, failures
            qa = future.result()
            attempts += 1
            if qa:
                q_key = qa["question"].strip().lower()
                if q_key in seen_questions:
                    failures += 1
                    return
                if count < target:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    f.flush()
                    seen_questions.add(q_key)
                    count += 1
                    pbar.update(1)
                    return
            failures += 1

        for pass_idx in range(max_passes):
            pass_seed = seed + pass_idx
            pass_start_count = count
            for record in iter_traffic_corpus_records(shuffle_seed=pass_seed):
                if count >= target:
                    break
                record_seed = _stable_int(record.get("source", "unknown")) ^ pass_seed
                pending.append(
                    pool.submit(
                        _process,
                        record,
                        system_prompt,
                        min_chunk,
                        record_seed,
                        model,
                        backend,
                        openrouter_model,
                        timeout,
                        num_predict,
                        num_ctx,
                        num_gpu,
                    )
                )
                if len(pending) >= max_workers:
                    _collect(pending.pop(0))

            while pending:
                if count >= target:
                    pending.clear()
                    break
                _collect(pending.pop(0))

            elapsed = max(time.time() - start_time, 1e-6)
            rate = count / elapsed * 60
            ok = attempts - failures
            pbar.set_postfix(speed=f"{rate:.1f}p/m", ok=f"{ok}/{attempts}")

            if count >= target:
                break
            if count == pass_start_count:
                print(f"{phase_name}: corpus pass {pass_idx + 1} produced no new pairs.")
                break

        pbar.close()

    ok = attempts - failures
    print(f"{phase_name} done: {count} total | attempts={attempts} ok={ok} failures={failures}")


def _generate_negatives(
    splitter: RecursiveCharacterTextSplitter,
    seen_questions: set[str],
    final_target: int,
) -> None:
    count, _ = _read_existing_questions()
    if count >= final_target:
        print(f"Negatives: already at {count}/{final_target}, skipping.")
        return

    positives: list[dict] = []
    with _get_output_path().open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("corpus") == "local_text":
                    positives.append(item)

    source_chunks: dict[str, list[dict]] = {}
    for record in iter_traffic_corpus_records(shuffle_seed=NEG_SEED):
        chunks = []
        for chunk in splitter.split_text(record["text"]):
            if len(chunk) >= 300 and not _contains_forbidden_legacy(chunk):
                chunks.append({"text": chunk, "article": infer_article(chunk)})
        source_chunks[record["doc_id"]] = chunks

    all_sources = [src for src, chunks in source_chunks.items() if chunks]
    if len(all_sources) < 2:
        print("Negatives skipped: need at least two source documents.")
        return

    needed = final_target - count
    rng = random.Random(NEG_SEED)
    rng.shuffle(positives)
    added = 0

    with _get_output_path().open("a", encoding="utf-8") as f, tqdm(
        total=needed, desc="Negatives", unit="pair"
    ) as pbar:
        for qa in positives:
            if added >= needed:
                break
            other_sources = [src for src in all_sources if src != qa.get("doc_id")]
            if not other_sources:
                continue
            chosen_src = rng.choice(other_sources)
            chosen_chunk = rng.choice(source_chunks[chosen_src])
            q_key = qa["question"].strip().lower()
            if q_key in seen_questions:
                # Reusing the same question with a negative context teaches refusal.
                pass
            neg = {
                "question": qa["question"],
                "answer": REFUSAL_ANSWER,
                "context": chosen_chunk["text"],
                "article": chosen_chunk["article"],
                "source": chosen_src,
                "doc_id": chosen_src,
                "title": "",
                "authority": "",
                "source_path": "",
                "corpus": "negative",
                "source_policy": KB_SOURCE_POLICY,
            }
            f.write(json.dumps(neg, ensure_ascii=False) + "\n")
            f.flush()
            added += 1
            pbar.update(1)

    print(f"Negatives done: {added} added -> {count + added} total")


def _phase_target_map(profile: str | None, target_cap: int | None) -> tuple[dict[str, int], int]:
    targets = {key: target for key, _, _, _, _, target in PHASE_CONFIGS}
    negative_target = NEG_FINAL_TARGET

    if profile == "smoke":
        for key, target in SMOKE_PHASE_TARGETS.items():
            targets[key] = target
        negative_target = SMOKE_NEGATIVE_TARGET

    if target_cap is not None:
        if target_cap <= 0:
            raise ValueError("--target-cap must be > 0")
        for key in list(targets):
            targets[key] = min(targets[key], target_cap)
        negative_target = min(negative_target, target_cap)

    return targets, negative_target


def generate(
    force: bool = False,
    phases: list[str] | None = None,
    profile: str | None = None,
    target_cap: int | None = None,
    phase_size: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_passes: int = DEFAULT_MAX_PHASE_PASSES,
    negatives: bool = True,
    model: str = MODEL,
    backend: str = "ollama",
    openrouter_model: str = OPENROUTER_MODEL,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    output_path: Path | None = None,
) -> None:
    global _CUSTOM_OUTPUT_PATH
    _CUSTOM_OUTPUT_PATH = output_path

    out = _get_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if force and out.exists():
        out.unlink()
    if max_workers <= 0:
        raise ValueError("--max-workers must be > 0")
    if max_passes <= 0:
        raise ValueError("--max-passes must be > 0")
    if phase_size is not None and phase_size <= 0:
        raise ValueError("--phase-size must be > 0")

    _, seen_questions = _read_existing_questions()
    neg_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    selected_phases = set(phases or [key for key, *_ in PHASE_CONFIGS])
    phase_targets, negative_target = _phase_target_map(profile, target_cap)

    print(
        f"Config: profile={profile or 'full'} phases={sorted(selected_phases)} "
        f"max_workers={max_workers} max_passes={max_passes} "
        f"backend={backend} model={model if backend == 'ollama' else openrouter_model} "
        f"timeout={timeout}s num_predict={num_predict} num_ctx={num_ctx} "
        f"num_gpu={num_gpu}"
    )
    for phase_key, phase_name, prompt, seed, min_chunk, _default_target in PHASE_CONFIGS:
        if phase_key not in selected_phases:
            print(f"{phase_name}: skipped by --phases")
            continue
        target = phase_targets[phase_key]
        if phase_size is not None:
            count, _ = _read_existing_questions()
            target = count + phase_size
        _run_phase(
            phase_key,
            phase_name,
            prompt,
            seed,
            min_chunk,
            target,
            seen_questions,
            max_workers=max_workers,
            max_passes=max_passes,
            model=model,
            backend=backend,
            openrouter_model=openrouter_model,
            timeout=timeout,
            num_predict=num_predict,
            num_ctx=num_ctx,
            num_gpu=num_gpu,
        )

    if negatives:
        _generate_negatives(neg_splitter, seen_questions, negative_target)
    else:
        print("Negatives: skipped by --no-negatives")

    total, _ = _read_existing_questions()
    print(f"\nAll done: {total} QA pairs -> {_get_output_path()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Delete existing QA data before generation.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="full",
        help="Use a smaller preset target set for quick QA generation.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=[key for key, *_ in PHASE_CONFIGS],
        help="Only run the selected QA generation phases.",
    )
    parser.add_argument(
        "--target-cap",
        type=int,
        default=None,
        help="Cap every cumulative phase target to this value.",
    )
    parser.add_argument(
        "--phase-size",
        type=int,
        default=None,
        help="Generate this many new QA pairs per selected phase.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of concurrent Ollama requests.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=DEFAULT_MAX_PHASE_PASSES,
        help="Maximum corpus passes per phase.",
    )
    parser.add_argument(
        "--no-negatives",
        action="store_true",
        help="Skip negative/refusal sample generation.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Ollama model name to use for QA generation.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openrouter"],
        default="ollama" if not OPENROUTER_API_KEY else "openrouter",
        help="Backend for QA generation (ollama=local, openrouter=cloud).",
    )
    parser.add_argument(
        "--openrouter-model",
        default=OPENROUTER_MODEL,
        help="OpenRouter model name (default: google/gemini-2.0-flash-001).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT,
        help="Timeout in seconds for each Ollama request.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help="Maximum tokens generated by Ollama per request.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="Ollama context window for generation.",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=DEFAULT_NUM_GPU,
        help="Number of model layers to place on GPU; use a large value to request full offload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write QA pairs to this file instead of the default qa_pairs_traffic.jsonl.",
    )
    args = parser.parse_args()
    generate(
        force=args.force,
        phases=args.phases,
        profile=None if args.profile == "full" else args.profile,
        target_cap=args.target_cap,
        phase_size=args.phase_size,
        max_workers=args.max_workers,
        max_passes=args.max_passes,
        negatives=not args.no_negatives,
        model=args.model,
        backend=args.backend,
        openrouter_model=args.openrouter_model,
        timeout=args.timeout,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        num_gpu=args.num_gpu,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
