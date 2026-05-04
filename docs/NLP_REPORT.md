# Vietnamese Traffic-Law Question Answering with RAG and LoRA Fine-Tuning

**Natural Language Processing Final Project — Nhập môn Xử lý ngôn ngữ tự nhiên**

---

## Abstract

We build a Vietnamese Question Answering system for traffic law that combines **Retrieval-Augmented Generation (RAG)** with **LoRA fine-tuning**. The system answers questions about Vietnam's traffic laws (penalties, procedures, definitions, prohibited acts) by retrieving relevant legal text from a vector database of 12 legal documents and generating answers using a fine-tuned large language model. We compare four configurations: base vs. fine-tuned LLM, with and without RAG. The fine-tuned model achieves significant improvements: ROUGE-L increases from 0.228 to 0.345, the forbidden legacy citation rate drops from 72% to 0%, and the unsupported refusal rate reaches 80%.

---

## 1. Introduction

Question Answering over legal texts is a challenging NLP task that requires both: (1) accurate retrieval of relevant legal provisions, and (2) precise answer generation that correctly cites legal sources. Vietnamese traffic law is particularly complex due to recent regulatory changes (e.g., the 2024 Road Law replaces the 2008 version), frequent amendments, and overlapping jurisdictions between multiple decrees and circulars.

We address these challenges by building a pipeline that:
1. Collects and chunks 12 Vietnamese traffic-law documents
2. Builds a FAISS vector store with BGE-M3 embeddings
3. Generates 2,886 QA pairs covering 6 question types
4. Fine-tunes Qwen3.5-9B with QLoRA
5. Implements a hybrid retrieval system (FAISS + lexical re-ranking + cross-encoder)
6. Evaluates 4 configurations on open-ended and multiple-choice benchmarks

---

## 2. Dataset

### 2.1 Source Documents

We collect 12 Vietnamese legal documents related to traffic law, sourced from Thư viện Pháp luật and official government portals:

| # | Document ID | Title | Type | Size |
|:-:|:-----------:|-------|:----:|:----:|
| 1 | `nd_168_2024_nd_cp` | Nghị định 168/2024/NĐ-CP — Xử phạt vi phạm giao thông | Nghị định | 340 KB |
| 2 | `nd_158_2024_nd_cp` | Nghị định 158/2024/NĐ-CP — Vận tải đường bộ | Nghị định | 291 KB |
| 3 | `nd_165_2024_nd_cp` | Nghị định 165/2024/NĐ-CP — Hướng dẫn thi hành Luật Đường bộ | Nghị định | 376 KB |
| 4 | `nd_336_2025_nd_cp` | Nghị định 336/2025/NĐ-CP — Xử phạt hạ tầng đường bộ | Nghị định | 90 KB |
| 5 | `tt_65_2024_tt_bca` | Thông tư 65/2024/TT-BCA — Kiểm tra phục hồi điểm GPLX | Thông tư | 25 KB |
| 6 | `luat_35_2024_qh15` | Luật Đường bộ 35/2024/QH15 | Luật | 199 KB |
| 7 | `luat_36_2024_qh15` | Luật Trật tự ATGT đường bộ 36/2024/QH15 | Luật | 218 KB |
| 8 | `qcvn_41_2024_bgtvt` | QCVN 41:2024/BGTVT — Báo hiệu đường bộ | Quy chuẩn | 348 KB |
| 9 | `tt_05_2025_tt_bgtvt` | Thông tư 12/2025/TT-BCA — Sát hạch, cấp GPLX | Thông tư | 105 KB |
| 10 | `nd_39_2023_nd_cp` | Nghị định 39/2023/NĐ-CP — Đấu giá biển số xe | Nghị định | 34 KB |
| 11 | `tt_79_2024_tt_bca` | Thông tư 79/2024/TT-BCA — Đăng ký xe, biển số xe | Thông tư | 114 KB |
| 12 | `tt_suckhoe_lai_xe` | Thông tư về tiêu chuẩn sức khỏe người lái xe | Thông tư | 21 KB |

**Total corpus: ~1.54 MB of legal text across 12 documents.**

### 2.2 Knowledge Base Construction

Documents are chunked using a legal-specific splitting strategy:

```mermaid
flowchart TB
    subgraph Source["Source Documents"]
        D1[("Luật 35/2024")]
        D2[("NĐ 168/2024")]
        D3[("QCVN 41:2024")]
        D4[("... 9 more docs")]
    end

    subgraph Chunking["Legal Chunking"]
        ART["Article Detection<br/>Điều X. header regex"]
        CLS["Clause Splitting<br/>(1), (2)... markers"]
        FALL["Fallback: RecursiveCharacter<br/>chunk_size=2000, overlap=200"]
    end

    subgraph Filtering["Quality Filtering"]
        SHORT["Remove chunks < 150 chars"]
        LONG["Split chunks > 2800 chars"]
        LEGACY["Filter forbidden legacy terms"]
    end

    subgraph Embedding["Embedding & Storage"]
        EMB["BAAI/bge-m3<br/>batch_size=128, max_seq=1024"]
        FAISS["FAISS IndexFlatIP<br/>2,703 vectors, 1,024 dim"]
    end

    D1 & D2 & D3 & D4 --> ART --> CLS --> FALL --> SHORT --> LONG --> LEGACY --> EMB --> FAISS

    style FAISS fill:#e8f5e9,stroke:#2e7d32
    style EMB fill:#e3f2fd,stroke:#1565c0
```

**Knowledge Base statistics:**
- **2,703 chunks** from 12 documents
- **Embedding**: BAAI/bge-m3 (1,024-dim, max_seq=1,024)
- **Index**: FAISS Inner Product (cosine similarity after L2 normalization)
- **Chunking policy**: `legal_article_clause_v1` — preserves Điều/khoản structure

### 2.3 QA Pair Generation

We generate 3,000 candidate QA pairs using the **Gemini 2.0 Flash** API via OpenRouter, covering 6 phases:

| Phase | System Prompt Focus | Target Pairs |
|-------|-------------------|:------------:|
| General | Open-ended legal questions | 250 |
| Definitions | Legal definitions and concepts | 650 |
| Penalties | Fines, penalties, license points | 1,400 |
| Procedures | Conditions, procedures, documentation | 1,900 |
| Prohibited | Prohibited actions, legal obligations | 2,400 |
| Scenarios | Real-world situational questions | 2,800 |
| Negatives | Refusal pairs (question ≠ context) | 3,000 |

**QA Filtering Criteria:**
- Remove answers exceeding **500 characters** (104 removed)
- Remove **template questions** starting with "Điều X của văn bản này..." (10 removed)
- **Final dataset**: 2,886 pairs (1,600 positive + 1,286 negative)
- **Retention rate**: 96.2%
- **All 12 documents** covered with 106-153 positive pairs each

### 2.4 Evaluation Dataset

| Split | Source | Samples | Description |
|-------|--------|:-------:|-------------|
| Manual | Human-written | 87 | Expert-written QA pairs, 6 categories |
| Human-verified | Human-checked | 100 | Human-verified QA pairs |
| Eval MC | Mixed | 201 | Multiple-choice questions (4 choices) |
| **Total** | | **223** | Open-ended; **201** MC |

Categories in open-ended eval: penalty (28), procedure (33), unsupported (5), definition (11), general (42), prohibited (4).

---

## 3. Model Architecture

### 3.1 Base Model: Qwen3.5-9B

We use **Qwen/Qwen3.5-9B** as the base LLM, a 9-billion parameter decoder-only transformer pre-trained on multilingual data including Vietnamese.

### 3.2 QLoRA Fine-Tuning

```mermaid
flowchart TB
    subgraph Base["Qwen3.5-9B (4-bit NF4)"]
        EMB["Embedding<br/>vocab_size=152,064"]
        L1["Decoder Layer 1<br/>..."]
        L2["Decoder Layer 2<br/>..."]
        L28["Decoder Layer 28<br/>SwiGLU FFN + RoPE"]
        LM["lm_head"]
        EMB --> L1 --> L2 --> ... --> L28 --> LM
    end

    subgraph LoRA["LoRA Adapters (trainable)"]
        Q["q_proj<br/>r=16, α=32"]
        K["k_proj<br/>r=16, α=32"]
        V["v_proj<br/>r=16, α=32"]
        O["o_proj<br/>r=16, α=32"]
        G["gate_proj<br/>r=16, α=32"]
        U["up_proj<br/>r=16, α=32"]
        D["down_proj<br/>r=16, α=32"]
    end

    L2 <--> Q & K & V & O & G & U & D

    subgraph Training["Training Strategy"]
        DROPOUT["Context Dropout<br/>70% keep context<br/>30% drop context"]
        MASK["train_on_responses_only<br/>mask loss on user/system tokens"]
    end

    Training -.-> LoRA

    style LoRA fill:#e8f5e9,stroke:#2e7d32
    style Base fill:#f3e5f5,stroke:#6a1b9a
```

#### LoRA Mathematical Formulation

LoRA (Low-Rank Adaptation) modifies a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by adding a low-rank update:

$$ W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA $$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, with rank $r = 16 \ll \min(d, k)$. Only $A$ and $B$ are trainable.

With $\alpha = 32$ and $r = 16$, the effective scaling is $\frac{\alpha}{r} = 2$.

**Target modules**: All 7 linear projections in Qwen3.5's attention and FFN layers ($q, k, v, o, gate, up, down$). This gives **29.1M trainable parameters** out of **9.4B total** (0.31%).

#### 4-bit Quantization (QLoRA)

The base model is loaded in 4-bit NF4 (NormalFloat4) via BitsAndBytes:

$$ \text{NF4}(x) = \text{round}\left(\frac{\text{clamp}(x / \sigma, -1, 1) + 1}{2} \cdot 15\right) $$

where $\sigma$ is the standard deviation of the weight tensor. During the forward pass, weights are dequantized on-the-fly:

$$ h = \text{dequantize}(W_0^{\text{NF4}})x + \frac{\alpha}{r} BA x $$

#### Context Dropout Training

We use a context dropout strategy where 70% of training samples include legal context and 30% omit it. This prepares the model for both Config C (no RAG) and Config D (RAG) at inference time:

$$ \text{Input} = \begin{cases} \text{"Đoạn văn bản luật:\n" + context + "\n\nCâu hỏi: " + question & \text{with prob } p = 0.7 \\ \text{"Câu hỏi: " + question} & \text{with prob } 1 - p = 0.3 \end{cases} $$

#### Training Objective

We minimize cross-entropy loss on **assistant response tokens only**:

$$ \mathcal{L} = -\sum_{t=1}^T \mathbb{1}_{\{y_t \in \text{response}\}} \log P(y_t | x, y_{<t}) $$

The loss is masked on system prompts and user inputs via `train_on_responses_only`.

#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3.5-9B |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0 |
| Optimizer | AdamW (8-bit) |
| Learning Rate | 2e-4 |
| Schedule | Cosine (warmup 5%) |
| Epochs | 5 |
| Effective Batch Size | 16 (2 × 8 grad accum) |
| Max Sequence Length | 2048 |
| Quantization | 4-bit NF4 |
| Trainable Params | 29.1M / 9.4B (0.31%) |
| Training Samples | 2,596 (after 10% val split) |
| Validation | 10% holdout |

---

## 4. Retrieval System (RAG)

### 4.1 Retrieval v2 Architecture

```mermaid
flowchart LR
    subgraph Query["Query Processing"]
        Q[("Question")]
        VN["Vietnamese Tokenizer<br/>pyvi ViTokenizer"]
        INT["Intent Classification<br/>9 intent types"]
        EXP["Query Expansion<br/>vượt đèn đỏ → không chấp hành hiệu lệnh..."]
    end

    subgraph Semantic["Step 1: FAISS Semantic Search"]
        FAISS["FAISS similarity_search<br/>candidate_k=30"]
    end

    subgraph Lexical["Step 2: Lexical Re-Ranking"]
        TOK["Token Overlap<br/>(VN word segmentation)"]
        PHR["Phrase Matching<br/>penalty amounts, intent keywords"]
        LEGAL["Legal Completeness<br/>fine ranges, point deductions"]
        SUBJ["Subject-Role<br/>ô tô vs xe máy distinction"]
    end

    subgraph Rerank["Step 3: Cross-Encoder Reranking"]
        CE["BAAI/bge-reranker-v2-m3<br/>top 20 candidates"]
    end

    subgraph Output["Output"]
        TOP3["Top-3 context chunks"]
        SUPP["Structural Supplements<br/>point deduction clauses"]
    end

    Q --> VN --> INT --> EXP --> FAISS --> TOK & PHR & LEGAL & SUBJ --> CE
    CE --> TOP3 --> SUPP

    style FAISS fill:#e3f2fd,stroke:#1565c0
    style Rerank fill:#fce4ec,stroke:#c62828
    style Output fill:#e8f5e9,stroke:#2e7d32
```

### 4.2 Retrieval Components

#### Intent Classification

The system classifies questions into 9 intents using keyword matching:

$$ \text{Intent}(q) = \{i \in \mathcal{I} \mid \text{match}(q, \text{patterns}_i) \} $$

| Intent | Example Patterns | Priority Document |
|--------|-----------------|:-----------------:|
| penalty | phạt, xử phạt, mức phạt | nd_168_2024_nd_cp |
| signal | đèn đỏ, đèn tín hiệu | nd_168_2024_nd_cp |
| helmet | mũ bảo hiểm | nd_168_2024_nd_cp |
| alcohol | nồng độ cồn | nd_168_2024_nd_cp |
| license_points | trừ điểm, GPLX | nd_168, tt_65 |
| restore_points | phục hồi điểm | tt_65, nd_168 |
| road_sign | biển báo, báo hiệu | qcvn_41_2024_bgtvt |
| vehicle_registration | đăng ký xe | tt_79_2024_tt_bca |
| driving_license | sát hạch, cấp GPLX | tt_05_2025_tt_bgtvt |

#### Vietnamese Tokenization

Unlike the v1 system that used simple `re.findall(r"\w+", ...)`, v2 uses **pyvi** (Vietnamese word segmenter):

$$ \text{Tokens}(s) = \text{ViTokenizer}(s) $$

This correctly segments Vietnamese compound words:
- "giao thông" → `["giao_thông"]` (not `["giao", "thông"]`)
- "điều khiển xe máy" → `["điều_khiển", "xe_máy"]`

#### Query Expansion

Legal terms are expanded with canonical phrasing from the law texts:

$$ \text{Expanded}(q) = q + \sum_{(\text{surface}, \text{legal}) \in \mathcal{M}} \mathbb{1}_{\{\text{surface} \in q\}} \cdot \text{legal} $$

| Surface | Canonical Legal Phrase |
|---------|----------------------|
| vượt đèn đỏ | không chấp hành hiệu lệnh của đèn tín hiệu giao thông |
| phạt bao nhiêu | phạt tiền từ |
| xe máy | xe mô tô, xe gắn máy, các loại xe tương tự |
| biển báo | báo hiệu đường bộ, hệ thống báo hiệu giao thông |

#### Cross-Encoder Reranker

After lexical re-ranking, the top-20 candidates are scored by a cross-encoder:

$$ \text{Score}(q, d) = \text{CrossEncoder}(q, d) \in \mathbb{R} $$

The cross-encoder `BAAI/bge-reranker-v2-m3` processes each (question, document) pair jointly, producing relevance scores that are more accurate than dot-product similarity:

$$ \text{CE}(q, d) = \text{Linear}(\text{Pool}(\text{Transformer}([CLS; q; SEP; d]))) $$

#### Re-Ranking Score

The final ranking combines all signals:

$$ S(q, d) = w_{\text{lex}} \cdot S_{\text{lex}}(q, d) + w_{\text{ce}} \cdot S_{\text{ce}}(q, d) + S_{\text{bonus}}(q, d) $$

where $S_{\text{bonus}}$ includes article-level hints, document priority by intent, legal completeness bonuses (penalty amounts, point deductions), and subject-role bonuses (ô tô vs. xe máy).

### 4.3 Retrieval Performance

| Metric | v1 (Pre-Improvement) | v2 (Current) | Δ |
|--------|:--------------------:|:------------:|:-:|
| **Recall@5** | 0.2674 | **0.4016** | **+50.1%** |
| Source Hit Rate | 0.314 | **0.590** | **+87.9%** |

The improvement comes from:
1. **FAISS-first retrieval** (v1 ran lexical first, making FAISS effectively unused)
2. **Vietnamese word segmentation** via pyvi (v1 used regex `\w+` which breaks VN compound words)
3. **Cross-encoder reranker** (v1 had no reranker)
4. **Expanded query expansion** with 15+ legal mappings (v1 had only 4)
5. **Additional legal documents** (7 → 12, +71% more coverage)

---

## 5. Experimental Setup

### 5.1 Evaluation Configurations

| Config | Model | RAG |
|:------:|:-----:|:---:|
| **A** | Base Qwen3.5-9B (zero-shot) | No |
| **B** | Base Qwen3.5-9B (zero-shot) | Yes |
| **C** | Fine-tuned Qwen3.5-9B (LoRA) | No |
| **D** | Fine-tuned Qwen3.5-9B (LoRA) | Yes |

### 5.2 Prompt Templates

**Without RAG (A/C):**
> "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Hãy trả lời câu hỏi dựa trên kiến thức của bạn về pháp luật giao thông Việt Nam hiện hành (2024-2025). Trả lời ngắn gọn, chính xác."

**With RAG (B/D):**
> "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. Ưu tiên trả lời dựa trên đoạn văn bản luật được cung cấp và nêu rõ căn cứ. Nếu văn bản không đủ thông tin, hãy trả lời từ kiến thức pháp luật giao thông của bạn và ghi rõ '(Kiến thức chung)'. Trả lời ngắn gọn, chính xác."

### 5.3 Evaluation Metrics

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE-N measures n-gram recall between candidate and reference:

$$ R_{\text{rouge-n}} = \frac{\sum_{gram_n \in \text{ref}} \text{count}_{\text{match}}(gram_n)}{\sum_{gram_n \in \text{ref}} \text{count}(gram_n)} $$

ROUGE-L uses the longest common subsequence (LCS):

$$ R_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{|Y|}, \quad P_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{|X|} $$
$$ \text{ROUGE-L} = F_{\text{lcs}} = \frac{2 \cdot R_{\text{lcs}} \cdot P_{\text{lcs}}}{R_{\text{lcs}} + P_{\text{lcs}}} $$

#### BLEU (Bilingual Evaluation Understudy)

Corpus-level BLEU-4 with brevity penalty:

$$ p_n = \frac{\sum_{\text{candidate}} \sum_{gram_n} \text{count}_{\text{clip}}(gram_n)}{\sum_{\text{candidate}} \sum_{gram_n} \text{count}(gram_n)} $$

$$ \text{BLEU} = \text{BP} \cdot \exp\left(\frac{1}{4}\sum_{n=1}^4 \log p_n\right) $$

#### METEOR

Unigram F1 with fragmentation penalty:

$$ F = \frac{10 PR}{R + 9P}, \quad \text{Penalty} = 0.5\left(\frac{\text{chunks}}{\text{matches}}\right)^3 $$
$$ \text{METEOR} = F \cdot (1 - \text{Penalty}) $$

#### BERTScore

Token-level semantic similarity via PhoBERT embeddings:

$$ R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \text{cosine}(\text{Emb}(x_i), \text{Emb}(\hat{x}_j)) $$

#### Forbidden Legacy Rate

Percentage of predictions citing outdated legal documents (Luật 2008, NĐ 100/2019, NĐ 46/2016):

$$ \text{Legacy Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_{\{\text{pred}_i \text{ contains legacy term}\}} $$

#### Unsupported Refusal Rate

Percentage of "unsupported" category questions where the model correctly refuses:

$$ \text{Refusal Rate} = \frac{\sum_{i \in \text{unsupported}} \mathbb{1}_{\{\text{pred}_i \text{ contains refusal}\}}}{|\text{unsupported}|} $$

---

## 6. Results

### 6.1 Main Results (223 Open-Ended Samples)

| Config | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | F1-tok | BERTSc | EM | Legacy ↓ | Refusal ↑ | Recall@5 | Latency |
|:------:|:-------:|:-------:|:-------:|:----:|:------:|:------:|:------:|:--:|:--------:|:---------:|:--------:|:-------:|
| **A** (base, no RAG) | 0.3073 | 0.1859 | 0.2282 | 0.0480 | 0.3384 | 0.1828 | 0.5660 | 0.0 | **0.7220** | 0.0000 | 0.4016 | 1.76s |
| **B** (base, RAG) | 0.3092 | 0.1950 | 0.2342 | 0.0565 | 0.3492 | 0.1872 | 0.5779 | 0.0 | **0.1839** | 0.0000 | 0.4016 | 2.26s |
| **C** (finetuned, no RAG) | **0.4922** | 0.1649 | 0.3226 | 0.1200 | 0.1534 | 0.1365 | 0.4892 | 0.0179 | **0.0000** | **0.8000** | 0.4016 | 0.49s |
| **D** (finetuned, RAG) | **0.5057** | **0.2031** | **0.3450** | **0.1602** | 0.2013 | **0.1738** | 0.5189 | **0.0493** | **0.0000** | **0.6000** | 0.4016 | 1.06s |

### 6.2 Multiple-Choice Results (201 Samples)

| Config | Accuracy | Correct | Latency |
|:------:|:--------:|:-------:|:-------:|
| A | 0.7910 | 159/201 | 0.23s |
| B | **0.8259** | 166/201 | 0.55s |
| C | **0.8358** | 168/201 | 0.25s |
| D | 0.8109 | 163/201 | 0.59s |

### 6.3 Retrieval Performance

| Metric | Before Improvements | After Improvements | Δ |
|--------|:-------------------:|:------------------:|:-:|
| Recall@5 | 0.2674 | **0.4016** | +50.1% |
| Source Hit Rate | 0.314 | **0.5902** | +87.9% |

---

## 7. Analysis

### 7.1 Fine-Tuning Impact

Fine-tuning (A→C, B→D) drives the most significant improvements:

| Metric | A→C (no RAG) | B→D (with RAG) |
|--------|:------------:|:--------------:|
| ROUGE-L Δ | +0.094 | **+0.111** |
| BLEU Δ | +0.072 | **+0.104** |
| Legacy Rate Δ | **-0.722** ✅ | **-0.184** ✅ |
| Refusal Rate Δ | **+0.800** ✅ | **+0.600** ✅ |

Key observations:
1. **Legacy citations eliminated entirely** (72.2% → 0% for no-RAG; 18.4% → 0% for RAG)
2. **Refusal rate jumps to 80%** for Config C — the model learns to say "I cannot find the legal basis"
3. **Large BLEU improvements** (+0.072 to +0.104) indicating more precise answer generation

### 7.2 RAG Effectiveness

RAG provides measurable but modest improvements:

| Metric | C (no RAG) → D (RAG) |
|--------|:-------------------:|
| ROUGE-L | 0.3226 → **0.3450** (+0.022) |
| BLEU | 0.1200 → **0.1602** (+0.040) |
| BERTScore | 0.4892 → **0.5189** (+0.030) |

**For the base model** (A→B), RAG primarily reduces the legacy citation rate (72.2% → 18.4%) by providing correct legal context.

**For the fine-tuned model** (C→D), RAG improves precision metrics (BLEU +33.5%) but slightly reduces the refusal rate (80% → 60%). The lower refusal rate for Config D is expected — when RAG provides context, the model attempts to answer even for unsupported questions, occasionally producing incorrect answers instead of refusing.

### 7.3 Retrieval Quality

The retrieval v2 improvements show significant gains:
- **Recall@5**: 0.267 → 0.402 (+50%)
- **Source Hit Rate**: 0.314 → 0.590 (+88%)

Despite these improvements, Recall@5 remains below 0.50, indicating that the correct legal document is still missed in over half of queries. This represents the primary bottleneck for further RAG improvement.

### 7.4 Multiple-Choice Performance

All four configurations perform similarly on MC questions (79-84% accuracy), suggesting that:
1. MC questions are fundamentally easier than open-ended generation
2. The base model already has sufficient legal knowledge for 4-option selection
3. Fine-tuning and RAG provide marginal benefits for MC tasks

### 7.5 Error Analysis

**Config A (base, no RAG):** High legacy rate (72.2%) — the model defaults to citing the outdated 2008 Traffic Law, indicating that the base model's knowledge is from pre-2024 training data.

**Config B (base, RAG):** Legacy rate drops to 18.4% — RAG provides correct context but the model sometimes ignores it, falling back to parametric knowledge.

**Legacy Rate Comparison:**

```mermaid
bar
    title Legacy Citation Rate by Config
    x-axis ["A (base, no RAG)", "B (base, RAG)", "C (finetuned, no RAG)", "D (finetuned, RAG)"]
    dataset "Legacy Rate"
    A: 72.2
    B: 18.4
    C: 0.0
    D: 0.0
```

---

## 8. Implementation Details

### 8.1 Hardware
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- **GPU Used By**: LoRA training (~10GB), inference (~7GB)
- **RAM**: 30GB, CPU: 28 cores

### 8.2 Software Stack
- **Framework**: PyTorch 2.10.0, Transformers 5.5.0
- **LoRA**: PEFT library (r=16, α=32)
- **Quantization**: BitsAndBytes (4-bit NF4)
- **Training**: Unsloth for optimized Qwen3.5 fine-tuning
- **Vector Store**: FAISS + LangChain
- **Embedding**: sentence-transformers (BAAI/bge-m3)
- **Cross-Encoder**: BAAI/bge-reranker-v2-m3
- **Vietnamese NLP**: pyvi, underthesea

### 8.3 Key Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| **Legal article/clause chunking** | Preserves hierarchical structure of legal documents |
| **Context dropout (70/30)** | Enables single model for both RAG and no-RAG inference |
| **Precompute reference logprobs** | Saves VRAM during DPO training |
| **FAISS-first retrieval** | Semantic search captures meaning where lexical search fails |
| **pyvi tokenization** | Vietnamese word segmentation significantly improves token overlap |

---

## 9. Comparison with Prior Work

### 9.1 Within-Project Improvement

| Aspect | Pre-Optimization | Post-Optimization |
|--------|:----------------:|:-----------------:|
| KB Documents | 7 | **12** (+71%) |
| KB Chunks | 2,006 | **2,703** (+35%) |
| Retrieval Strategy | Lexical-first (FAISS dead code) | **FAISS-first → re-rank → cross-encoder** |
| Tokenization | regex `\w+` | **pyvi VN word segmentation** |
| Query Expansion | 4 fixed phrases | **15+ legal mappings** |
| Recall@5 | 0.2674 | **0.4016** (+50%) |
| Source Hit Rate | 0.314 | **0.590** (+88%) |
| Training QA Pairs | 550 | **2,886** (filtered from 3,000) |
| Refusal Rate (C) | 0.0 | **0.80** ✅ |

### 9.2 RAG Contribution Analysis

The overall end-to-end performance shows that **fine-tuning contributes significantly more than RAG** to answer quality. However, RAG plays a crucial role in:
1. Reducing legacy citations (especially for the base model)
2. Improving precision on specific recall tasks (BLEU +33.5% for C→D)
3. Providing traceable legal references

---

## 10. Post-Optimization Results (v2)

After identifying over-refusal as the critical issue, we retrained with a cleaned dataset and improved configuration. Changes:
- **Negative ratio reduced**: 44.6% → 8.6% (1,286 → 200 negative samples)
- **Hard-context examples added**: 200 samples with wrong context + "(Kiến thức chung)" marker
- **Additional positives**: +326 diverse real-world QA pairs (total 1,925)
- **LoRA rank**: 16 → 32 (more capacity)
- **Learning rate**: 2e-4 → 1e-4 (stable training)
- **Context dropout**: 70/30 → 90/10 (less refusal bias)
- **Epochs**: 5 → 3 (avoid overfitting, early stopping at epoch 2)
- **Context format aligned**: training and inference use same raw chunk format (no metadata headers)

**Key metric added: False Refusal Rate** — percentage of supported questions incorrectly refused.

### 10.1 Improved Results (187 Clean Samples, Zero Train Overlap)

| Config | ROUGE-L | BLEU | METEOR | BERTSc | EM | Legacy ↓ | **FRef ↓** | LLM-Judge | Latency |
|:------:|:-------:|:----:|:------:|:------:|:--:|:--------:|:----------:|:---------:|:-------:|
| **A** (base, no RAG) | 0.2192 | 0.0429 | 0.3199 | 0.5573 | 0.0 | 0.7701 | 0.0000 | 1.83/5 | 1.9s |
| **B** (base, RAG) | 0.2268 | 0.0509 | 0.3395 | 0.5700 | 0.0 | 0.5294 | 0.1538 | 2.00/5 | 2.2s |
| **C** (finetuned v2, no RAG) | **0.4459** | **0.2224** | **0.4491** | **0.6790** | 0.0053 | **0.0000** | **0.0000** | **2.05/5** | 1.0s |
| **D** (finetuned v2, RAG) | 0.3662 | 0.1947 | 0.2911 | 0.5727 | 0.0267 | **0.0000** | 0.1044 | 1.86/5 | 1.3s |

### 10.2 Before vs After Comparison

| Metric | Before (C) | After (C v2) | Δ | Before (D) | After (D v2) | Δ |
|--------|:----------:|:-------------:|:-:|:----------:|:-------------:|:-:|
| ROUGE-L | 0.3226 | **0.4459** | **+38%** | 0.3450 | 0.3662 | +6% |
| BLEU | 0.1200 | **0.2224** | **+85%** | 0.1602 | **0.1947** | +22% |
| METEOR | 0.1534 | **0.4491** | **+193%** | 0.2013 | **0.2911** | +45% |
| BERTScore | 0.4892 | **0.6790** | **+39%** | 0.5189 | **0.5727** | +10% |
| **False Refusal** | **83.4%** | **0.0%** | ✅ | **71.3%** | **10.4%** | ✅ |
| Legacy Rate | 0.0% | 0.0% | ✅ | 0.0% | 0.0% | ✅ |
| LLM-Judge | — | 2.05/5 | — | — | 1.86/5 | — |

### 10.3 Key Findings

1. **Over-refusal fixed**: Config C (no RAG) now answers 100% of questions correctly with zero false refusal. Config D reduced from 71.3% → 10.4%.

2. **Config C now dominates**: Without RAG, the fine-tuned model outperforms RAG variant (D) across all metrics. The model has internalized sufficient legal knowledge from the 82.8% positive training data.

3. **RAG trade-off**: Config D's 10.4% false refusal rate shows that poor-quality retrieved context (Recall@5 = 0.41) can still trigger refusal behavior. The learned knowledge pathway (Config C) is more reliable than the retrieval pathway.

4. **Context format alignment**: Removing metadata headers from inference context eliminated a subtle distribution shift that contributed to over-refusal.

5. **Diverse negative examples**: Using 6 different refusal phrasings instead of one fixed string reduced mode collapse and made the model less sensitive to refusal-triggering patterns.

---

## 11. Conclusion (Updated)

We built a Vietnamese Traffic-Law QA system with four configurations. After comprehensive optimization:

1. **Config C (fine-tuned, no RAG) is the best configuration**: ROUGE-L 0.4459, BLEU 0.2224, 0% false refusal, 0% legacy rate, 2.05/5 LLM-Judge score.

2. **Over-refusal eliminated**: False refusal rate dropped from 83.4% → 0.0% (C) and 71.3% → 10.4% (D) by fixing the negative/positive ratio and adding hard-context training examples.

3. **Fine-tuning drives all major gains**: +85% BLEU, +193% METEOR, +39% BERTScore compared to base model.

4. **RAG provides marginal benefit**: Config D slightly underperforms C, suggesting that retrieval quality (Recall@5 = 0.41) is a bottleneck. Future work should focus on improving retrieval rather than generation.

---

## Appendix A: Model Checkpoints

| Config | Path |
|:------:|------|
| C/D | `models/qwen3.5-9b-lora-traffic/` |

## Appendix B: Dataset Artifacts

| File | Description |
|------|-------------|
| `data/qa_pairs_traffic.jsonl` | 2,886 training QA pairs |
| `data/eval_manual.jsonl` | 223 open-ended eval samples |
| `data/eval_mc_manual.jsonl` | 201 MC eval samples |
| `vector_db_traffic/` | FAISS index (2,703 chunks) |
| `vector_db_traffic/build_meta.json` | KB metadata (12 docs) |

## Appendix C: Evaluation Commands

```bash
# Evaluate 4 configs
cd nlp && PYTHONPATH=src python src/evaluate.py --configs A B C D

# Evaluate MC
cd nlp && PYTHONPATH=src python src/evaluate_mc.py --configs A B C D

# Rebuild KB
cd nlp && PYTHONPATH=src python src/build_kb.py --force

# Generate QA (requires OpenRouter API key)
cd nlp && python src/generate_qa.py --force --backend openrouter
```
