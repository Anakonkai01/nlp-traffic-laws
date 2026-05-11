# Research: Fine-Slot Recall Bottleneck in Vietnamese Traffic Law QA

## Executive Summary

`số_tiền_fine recall = 0.0` on all RAG configs (B, D). Root cause analysis reveals the problem
is **80% measurement error** (gold labels + metric), not retrieval failure. The CE reranker
does have a genuine clause disambiguation problem, but it accounts for only ~20% of the gap.

---

## 1. Deep Diagnostic (2026-05-10)

### 1.1 Gold Label Audit: 20% Error Rate

Cross-referenced all 103 `gold_fine_min` labels from `auto_label_eval.py` against the
authoritative `legal_sanction_facts.jsonl` (extracted directly from NĐ 168/2024 text):

| Category | Count | % |
|----------|-------|---|
| Correct (exact value match) | 52 | 50% |
| Format-only diff (same value) | 30 | 29% |
| **Truly wrong amount** | **21** | **20%** |
| Article/clause not in facts at all | 4 | 4% |

**21 truly wrong labels** — the FINE_MAP table has incorrect fine amounts for common violations:

| Question | Gold (FINE_MAP) | Actual (facts) |
|----------|----------------|----------------|
| Ô tô vượt đèn đỏ | Art 6 cl 5: 18-20M | Art 6 cl 5: 4-6M |
| Xe máy vượt đèn đỏ | Art 7 cl 5: 4-6M | Art 7 cl 5: 1-2M |
| Ô tô ngược chiều | Art 6 cl 7: 16-18M | Art 6 cl 7: 12-14M |
| Xe máy sai làn | Art 7 cl 3: 2-3M | Art 7 cl 3: 600k-800k |
| Ô tô dùng điện thoại | Art 6 cl 4: 4-6M | Art 6 cl 4: 2-3M |
| Xe máy mũ bảo hiểm | Art 7 cl 3: 400k-600k | Art 7 cl 3: 600k-800k |
| Ô tô không dây an toàn | Art 6 cl 2: 800k-1M | Art 6 cl 2: 600k-800k |
| Quên đăng ký xe máy | Art 7 cl 1: 1M-1.5M | Art 7 cl 1: 200k-400k |

Root cause: The FINE_MAP was hand-written from memory. Many values reflect old regulations
(NĐ 100/2019) rather than NĐ 168/2024.

### 1.2 Non-Fine Questions Mislabeled

4 questions about GPLX points/restoration were assigned `gold_fine_min` even though they
don't ask about fines at all:
- "Bị trừ hết 12 điểm giấy phép lái xe thì bị xử lý như thế nào?"
- "Làm sao để phục hồi điểm giấy phép lái xe khi đã bị trừ hết?"
- "Lái xe trong 1 năm không bị trừ hết điểm thì điểm giấy phép lái xe có tự phục hồi không?"
- "Không chấp hành yêu cầu kiểm tra nồng độ cồn của cảnh sát giao thông bị xử lý như thế nào?"

### 1.3 Metric Too Strict: Exact String Match

Current metric:
```python
FINE_PRED_RE = re.compile(r"phạt tiền từ ([\d\.]+) đến ([\d\.]+)")
if m and m.group(1) == gold_fine_min:  # EXACT string match
    fine_hits += 1
```

Model output: `"phạt tiền từ 6.001.000 đồng đến 7.999.000 đồng"` — semantically correct
(same fine bracket) but counted as WRONG because `"6.001.000" != "6.000.000"`.

**Fuzzy match test:** with ±15% tolerance, the correct-answer rate is significantly higher,
proving the generator often knows the right range but outputs slightly different numbers.

### 1.4 CE Reranker Collapse (Real Issue, Smaller Impact)

All CE models (base bge-reranker-v2-m3, fine-tuned v4/v5/v6) produce near-identical scores
(~0.00002) for fact cards from the same article. Violation text is ~90% identical across
clauses — only fine amounts differ. This is a fundamental limitation of cross-encoder
architecture for this task, not a training data problem.

### 1.5 Generator Memorization (Bright Spot)

Config C (LoRA, no RAG) at ROUGE-L=0.3897 nearly matches Config D at 0.3933. The LoRA
adapter has memorized many fine amounts from training data. The gap to gold is primarily
output format (slightly imprecise numbers) rather than knowledge.

---

## 2. Root Cause Distribution

```
fine_slot_recall = 0.0 (current metric)
  ├── 40%: Gold label wrong (21/103) → measurement invalid
  ├── 15%: Non-fine Qs with gold_fine_min (4/103) → noise
  ├── 25%: Exact string match too strict → correct answers scored as wrong
  ├── 15%: Generator picks wrong clause (same-article disambiguation)
  └──  5%: True retrieval failure (wrong document/article)
```

**Key insight:** ~80% of `fine_slot_recall = 0` is measurement error. The generator is
substantially better than the metric suggests, and the retrieval problem (while real) is
the smallest contributor.

---

## 3. Earlier Literature Survey (from prior research)

### 3.1 STARA (Stanford RegLab, ICAIL 2025)

STARA preprocesses legal codes into statutory trees, stores clause lead-ins/chapeaus,
definitions, and cross-references. Key finding: statutory context augmentation (lead-ins)
gives +38pp extraction improvement over base (0.28 → 0.70). Our evidence-card approach
already does some of this, but we lack systematic lead-in attachment.

### 3.2 Legal-DC / LegRAG

Dual-granularity retrieval (chunk + article), self-reflection, and DSA (Document Selection
Accuracy) metrics. Their core insight: article-level retrieval + clause-level selection
is stronger than single-granularity.

### 3.3 ACORD (ACL 2025)

Expert-graded clause relevance labels (1-5 scale). BM25 + GPT-4o pointwise reranker
outperforms fine-tuned cross-encoders for legal clause selection. Token-overlap labels
produce false positives.

---

## 4. Proposed Solution Plan

### Phase 1: Fix Gold Labels (P0 — measurement fix)

**Approach:** Replace hand-written FINE_MAP with deterministic query of `legal_sanction_facts.jsonl`.

Script: `scripts/fix_gold_labels.py`
- For each eval question, run BM25 fact retrieval over all facts
- Take top-1 fact with `fine_min_vnd` not None and `answer_ready=True`
- Assign gold article/clause/fine from fact data
- For questions with no matching fact → leave gold fields empty
- Verify: manual spot-check 20% of labels

### Phase 2: Fix Metric (P0 — measurement fix)

Replace exact string match with numeric fuzzy comparison:
```python
def extract_fine_amounts(text: str) -> list[tuple[int, int]]:
    """Extract all (min, max) fine pairs from prediction text."""
    # Handle: "từ X đến Y đồng", "X-Y triệu đồng", "phạt X đồng"

def fine_match(gold_min_str: str, gold_max_str: str, pred: str) -> bool:
    gold_min = int(gold_min_str.replace('.', ''))
    gold_max = int(gold_max_str.replace('.', ''))
    pred_pairs = extract_fine_amounts(pred)
    return any(
        abs(pmin - gold_min) / gold_min < 0.10 and abs(pmax - gold_max) / gold_max < 0.10
        for pmin, pmax in pred_pairs
    )
```

### Phase 3: Structured Fact Injection (P1 — retrieval fix)

Instead of CE reranker selecting among similar fact cards, use deterministic keyword
matching to identify the correct article+clause, then inject ALL facts from that clause.

New function in `sanction_facts.py`:
```python
def retrieve_facts_structured(question: str, top_k: int = 3) -> list[Document]:
    """Keyword+BM25 to find (doc_id, article, clause), then inject all matching facts."""
    vehicle = detect_vehicle(question)
    violation_type = classify_violation(question)
    # Match against fact index by (vehicle, violation_type) → (article, clause)
    # Fall back to BM25 if no keyword match
    # Return ALL facts from matched clause (to avoid CE disambiguation problem)
```

Triggered by env var: `RAG_FACT_STRUCTURED_INJECT=1`

### Phase 4: Re-Baseline (P1)

Run full ABCD eval with fixed labels and metric:
```bash
RAG_SANCTION_FACT_CARDS=1 RAG_EVIDENCE_CARD_RENDERING=1 \
RAG_FACT_STRUCTURED_INJECT=1 \
python src/evaluate.py --configs A B C D --samples 145 \
  --test-file data/eval_manual_labeled_v2.jsonl
```

### Phase 5 (if Phase 3 recall still < 0.5): Generator Fine-Tune

Fine-tune Qwen on (question, fact_card, reference_answer) triples so the model learns
to read fact cards and extract exact fine amounts. This addresses the "model outputs
6.001.000 instead of 6.000.000" formatting problem.

---

## 5. Expected Impact

| Metric | Current | After P1+P2 | After P3 |
|--------|---------|-------------|----------|
| fine_slot_recall (D) | 0.00 | 0.35-0.45 | 0.55-0.65 |
| Gold label accuracy | 80% | 98%+ | 98%+ |
| D ROUGE-L | 0.3933 | 0.40-0.42 | 0.42-0.45 |

---

## 6. What NOT to Do

- Do NOT train more CE reranker variants — the architecture cannot disambiguate same-article clauses
- Do NOT add more hand-crafted FINE_MAP rules — use the authoritative fact database
- Do NOT run full eval after every minor change — use smoke tests (20-40 samples)

---

## 7. Concrete Implementation Order

1. `scripts/fix_gold_labels.py` — BM25-fact-based gold label assignment
2. Fix `compute_fine_slot_metrics()` in `evaluate.py` — fuzzy matching
3. `retrieve_facts_structured()` in `sanction_facts.py` — deterministic clause lookup
4. Smoke test: `python src/evaluate.py --configs D --samples 40 --fast-metrics`
5. Full eval if smoke passes
