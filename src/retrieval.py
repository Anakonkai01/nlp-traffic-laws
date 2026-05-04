"""
Hybrid retrieval for Vietnamese traffic-law QA.

Architecture (v2 — redesigned for high Recall@k):
  1. FAISS semantic similarity search → top-N candidates (candidate_k=30)
  2. Lexical re-ranking (Vietnamese token overlap + phrase matching + legal bonuses)
  3. Cross-encoder reranker (BAAI/bge-reranker-v2-m3) on top reranked docs
  4. Deduplication + structural supplements (point deduction clauses)

Key fixes from v1:
  - FAISS ALWAYS runs first (v1 ran lexical first, semantic was dead code)
  - pyvi Vietnamese word segmentation (v1 used regex \w+ which breaks VN compound words)
  - Expanded query expansion: vượt đèn đỏ ↔ không chấp hành hiệu lệnh đèn tín hiệu etc.
  - Cross-encoder reranker for precision
"""

import re
from collections import Counter
from functools import lru_cache

import numpy as np


# ── Vocabulary ────────────────────────────────────────────────────────────────

TRAFFIC_SCOPE_TERMS = {
    "giao thông", "đường bộ", "xe", "ô tô", "xe máy", "mô tô", "gắn máy",
    "xe đạp", "đèn đỏ", "đèn vàng", "đèn tín hiệu", "biển báo",
    "nồng độ cồn", "mũ bảo hiểm", "giấy phép lái xe", "gplx",
    "trừ điểm", "tốc độ", "vượt xe", "dừng xe", "đỗ xe",
    "làn đường", "cao tốc", "xe ưu tiên", "tai nạn giao thông",
    "đăng ký xe", "biển số", "sát hạch", "bằng lái", "khám sức khỏe",
    "tiêu chuẩn sức khỏe", "vận tải", "phù hiệu", "đấu giá biển",
}

STOPWORDS = {
    "ai", "bao", "bị", "các", "cần", "cho", "của", "đang", "điều",
    "được", "gì", "hiện", "khi", "khiển", "là", "mức", "nay",
    "người", "nhiêu", "ra", "sao", "theo", "thì", "thế", "trong", "và", "với",
}

DOC_PRIORITY_BY_INTENT = {
    "penalty":            ["nd_168_2024_nd_cp"],
    "restore_points_partial": ["nd_168_2024_nd_cp"],
    "license_points":     ["nd_168_2024_nd_cp", "tt_65_2024_tt_bca"],
    "restore_points":     ["tt_65_2024_tt_bca", "nd_168_2024_nd_cp"],
    "vehicle_registration": ["tt_79_2024_tt_bca"],
    "driving_license":    ["tt_05_2025_tt_bgtvt", "tt_65_2024_tt_bca"],
    "road_sign":           ["qcvn_41_2024_bgtvt"],
    "auction_plate":       ["nd_39_2023_nd_cp"],
    "health":              ["tt_suckhoe_lai_xe"],
    "transport_operation": ["nd_158_2024_nd_cp"],
    "general_traffic_law": ["luat_36_2024_qh15", "luat_35_2024_qh15"],
}

ARTICLE_HINTS_BY_INTENT = {
    "car":       ["Điều 6."],
    "bike":      ["Điều 7."],
    "bicycle":   ["Điều 9."],
    "passenger": ["Điều 12.", "Điều 7."],
    "restore_points":  ["Điều 7.", "Điều 6.", "Điều 51."],
    "license_points":  ["Điều 50.", "Điều 51.", "Điều 7.", "Điều 6."],
    "helmet":    ["Điều 6."],
}

# Vietnamese → legal canonical phrasing
INTENT_PHRASES = {
    "signal":     "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "helmet":     "không đội mũ bảo hiểm cho người đi mô tô xe máy",
    "alcohol":    "nồng độ cồn trong máu hoặc hơi thở",
    "license_points": "trừ điểm giấy phép lái xe phục hồi điểm giấy phép lái xe",
    "health": "khám sức khỏe tiêu chuẩn sức khỏe người lái xe",
    "transport_operation": "kinh doanh vận tải phù hiệu hợp đồng vận tải",
    "general_traffic_law": "quy tắc giao thông đường bộ người tham gia giao thông đường bộ",
}

# Query → legal phrase expansion for semantic bridging
QUERY_TO_LEGAL = {
    "vượt đèn đỏ":    "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "vượt đèn vàng":  "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "đèn đỏ":         "đèn tín hiệu giao thông",
    "không đội mũ":   "không đội mũ bảo hiểm",
    "phạt bao nhiêu": "phạt tiền từ",
    "mức phạt":       "phạt tiền từ",
    "bị phạt":        "phạt tiền từ",
    "xe hơi":         "xe ô tô các loại xe tương tự xe ô tô",
    "xe máy":         "xe mô tô xe gắn máy các loại xe tương tự xe mô tô",
    "mô tô":          "xe mô tô xe gắn máy các loại xe tương tự xe mô tô",
    "xe đạp":         "xe đạp xe đạp máy xe thô sơ",
    "biển báo":       "báo hiệu đường bộ hệ thống báo hiệu giao thông",
    "biển số":        "biển số xe ô tô đăng ký xe biển số định danh",
    "đấu giá":        "đấu giá biển số xe ô tô tiền đặt trước người trúng đấu giá",
    "đăng ký xe":     "đăng ký xe cấp biển số xe hồ sơ đăng ký sang tên thu hồi đăng ký",
    "chuyển hướng":   "chuyển hướng xe không có tín hiệu báo hướng rẽ",
    "xi nhan":        "chuyển hướng xe không có tín hiệu báo hướng rẽ",
    "đỗ xe":          "đỗ xe trên đường không đúng quy định",
    "dừng xe":        "dừng xe trên đường không đúng quy định",
    "quá tốc độ":     "chạy quá tốc độ quy định phạt tiền từ",
    "vượt tốc độ":    "chạy quá tốc độ quy định phạt tiền từ",
    "vượt xe":        "vượt xe không đúng quy định",
    "đi ngược chiều": "đi ngược chiều của đường một chiều",
    "quay đầu":       "quay đầu xe không đúng quy định phạt tiền từ",
    "bằng lái":       "giấy phép lái xe cấp giấy phép lái xe sát hạch lái xe",
    "sát hạch":       "sát hạch lái xe cấp giấy phép lái xe hồ sơ cấp giấy phép lái xe",
    "cấp đổi":        "cấp đổi giấy phép lái xe hồ sơ cấp đổi giấy phép lái xe",
    "không bằng":     "không có giấy phép lái xe",
    "chở quá":        "chở quá số người quy định",
    "đèn chiếu sáng": "đèn chiếu sáng phía trước không đúng quy định",
    "còi xe":         "sử dụng còi xe không đúng quy định",
    "lạng lách":      "lạng lách đánh võng phạt tiền từ",
    "nồng độ cồn":    "nồng độ cồn trong máu hoặc hơi thở phạt tiền từ",
    "sơn xe":         "tự ý thay đổi màu sơn xe",
    "khám sức khỏe":  "khám sức khỏe tiêu chuẩn sức khỏe người lái xe",
    "tiêu chuẩn sức khỏe": "tiêu chuẩn sức khỏe người lái xe bệnh không được lái xe thị lực lái xe",
    "vạch kẻ đường":  "báo hiệu đường bộ vạch kẻ đường quy chuẩn báo hiệu đường bộ",
    "biển cấm":       "biển báo cấm quy chuẩn báo hiệu đường bộ",
    "biển hiệu lệnh": "biển hiệu lệnh quy chuẩn báo hiệu đường bộ",
    "biển chỉ dẫn":   "biển chỉ dẫn quy chuẩn báo hiệu đường bộ",
    "kinh doanh vận tải": "kinh doanh vận tải bằng xe ô tô phù hiệu hợp đồng vận tải",
    "phù hiệu":       "phù hiệu xe kinh doanh vận tải",
}

# ── Vietnamese tokenization ────────────────────────────────────────────────────

_pyvi_tk = None

def _vn_tokenizer():
    global _pyvi_tk
    if _pyvi_tk is None:
        from pyvi import ViTokenizer
        _pyvi_tk = ViTokenizer
    return _pyvi_tk

def _vn_tokenize(text: str) -> str:
    """Vietnamese word segmentation → returns space-separated normalized tokens."""
    try:
        return _vn_tokenizer().tokenize(text.lower())
    except Exception:
        return text.lower()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _vn_tokenize(text)).strip()


def _tokenize(text: str) -> list[str]:
    raw = _normalize(text)
    tokens = re.findall(r"\w+", raw)
    return [tok for tok in tokens if len(tok) >= 2 and tok not in STOPWORDS]


def _has_any(text: str, phrases: list[str]) -> bool:
    lowered = _normalize(text)
    return any(_normalize(phrase) in lowered for phrase in phrases)


# ── Intent classification ──────────────────────────────────────────────────────

def is_supported_traffic_question(question: str) -> bool:
    q = _normalize(question)
    return any(_normalize(term) in q for term in TRAFFIC_SCOPE_TERMS)


def classify_intents(question: str) -> list[str]:
    q = _normalize(question)
    intents: list[str] = []

    if _has_any(q, ["phạt", "xử phạt", "bao nhiêu", "mức phạt", "vượt đèn đỏ",
                     "không đội", "vượt đèn vàng"]):
        intents.append("penalty")
    if "mũ bảo hiểm" in q:
        intents.append("helmet")
    if _has_any(q, ["đèn đỏ", "đèn tín hiệu", "đèn giao thông", "đèn vàng"]):
        intents.append("signal")
    if _has_any(q, ["nồng độ cồn", "rượu", "bia"]):
        intents.append("alcohol")
    if _has_any(q, ["trừ điểm", "phục hồi điểm", "gplx", "giấy phép lái xe"]):
        intents.append("license_points")
    if "chưa bị trừ hết điểm" in q or ("12 điểm" in q and "phục hồi" in q):
        intents.append("restore_points_partial")
    if "phục hồi điểm" in q or ("trừ hết điểm" in q and "kiểm tra" in q):
        intents.append("restore_points")
    if _has_any(q, ["ô tô", "oto", "xe hơi"]):
        intents.append("car")
    if _has_any(q, ["xe máy", "mô tô", "gắn máy"]):
        intents.append("bike")
    if "xe đạp" in q:
        intents.append("bicycle")
    if _has_any(q, ["người được chở", "người ngồi sau", "ngồi sau", "hành khách"]):
        intents.append("passenger")
    if _has_any(q, ["biển báo", "báo hiệu", "vạch kẻ", "cọc tiêu", "biển hiệu",
                     "qcvn", "quy chuẩn"]):
        intents.append("road_sign")
    if _has_any(q, ["đăng ký xe", "cấp biển số", "biển số xe", "đăng kí xe"]):
        intents.append("vehicle_registration")
    if _has_any(q, ["đào tạo lái", "sát hạch", "thi bằng lái", "học lái xe",
                     "cấp giấy phép lái xe", "bằng lái"]):
        intents.append("driving_license")
    if _has_any(q, ["đấu giá biển số", "đấu giá biển"]):
        intents.append("auction_plate")
    if _has_any(q, ["khám sức khỏe", "tiêu chuẩn sức khỏe", "sức khỏe người lái", "thị lực", "bệnh"]):
        intents.append("health")
    if _has_any(q, ["kinh doanh vận tải", "vận tải", "phù hiệu", "hợp đồng vận tải"]):
        intents.append("transport_operation")
    if _has_any(q, ["quy tắc giao thông", "người tham gia giao thông", "luật đường bộ", "kết cấu hạ tầng"]):
        intents.append("general_traffic_law")

    return list(dict.fromkeys(intents))


# ── Query expansion ────────────────────────────────────────────────────────────

def expand_traffic_query(question: str) -> str:
    q = _normalize(question)
    intents = classify_intents(question)
    expansions: list[str] = []

    # Legal phrase expansions for specific intents
    for intent in ("signal", "helmet", "alcohol", "license_points", "health", "transport_operation", "general_traffic_law"):
        if intent in intents and intent in INTENT_PHRASES:
            expansions.append(INTENT_PHRASES[intent])

    if "car" in intents:
        expansions.append("xe ô tô các loại xe tương tự xe ô tô người điều khiển ô tô")
    if "bike" in intents:
        expansions.append("xe mô tô xe gắn máy các loại xe tương tự xe mô tô người điều khiển xe mô tô")
    if "bicycle" in intents:
        expansions.append("xe đạp xe đạp máy xe thô sơ")
    if "restore_points" in intents:
        expansions.append("kiểm tra lý thuyết kiến thức pháp luật theo mô phỏng trừ hết điểm phục hồi đủ 12 điểm")
    if "road_sign" in intents:
        expansions.append("báo hiệu đường bộ biển báo vạch kẻ đèn tín hiệu cọc tiêu")
    if "vehicle_registration" in intents:
        expansions.append("đăng ký xe cấp biển số xe hồ sơ đăng ký thủ tục đăng ký")
    if "driving_license" in intents:
        expansions.append("đào tạo sát hạch cấp giấy phép lái xe hồ sơ đăng ký kiểm tra")
    if "auction_plate" in intents:
        expansions.append("đấu giá biển số xe ô tô tiền đặt trước người trúng đấu giá")
    if "health" in intents:
        expansions.append("khám sức khỏe tiêu chuẩn sức khỏe người lái xe bệnh không được lái xe thị lực")
    if "transport_operation" in intents:
        expansions.append("kinh doanh vận tải bằng xe ô tô phù hiệu hợp đồng vận tải")
    if "general_traffic_law" in intents:
        expansions.append("quy tắc giao thông đường bộ người tham gia giao thông đường bộ")

    # Substring-level fallback expansions
    for surface, legal in QUERY_TO_LEGAL.items():
        if _normalize(surface) in q and legal not in "\n".join(expansions):
            expansions.append(legal)

    if not expansions:
        return question
    return question + "\n" + "\n".join(dict.fromkeys(expansions))


# ── Scoring heuristics ─────────────────────────────────────────────────────────

def _doc_priority(question: str, doc_id: str) -> int:
    bonus = 0
    for intent in classify_intents(question):
        for rank, preferred in enumerate(DOC_PRIORITY_BY_INTENT.get(intent, [])):
            if preferred == doc_id:
                bonus += 25 - rank * 5
    return bonus


def _phrase_score(question: str, text: str) -> int:
    q = _normalize(question)
    t = _normalize(text)
    score = 0

    for phrase in INTENT_PHRASES.values():
        if phrase in t and any(tok in q for tok in _tokenize(phrase)):
            score += 20
    if _has_any(q, ["mũ bảo hiểm"]) and "mũ bảo hiểm" in t:
        score += 30
    if _has_any(q, ["đèn đỏ", "đèn tín hiệu", "đèn giao thông"]) and INTENT_PHRASES["signal"] in t:
        score += 35
    if _has_any(q, ["nồng độ cồn", "rượu", "bia"]) and "nồng độ cồn" in t:
        score += 35
    if _has_any(q, ["phục hồi điểm", "trừ hết điểm"]) and _has_any(
        t, ["phục hồi điểm", "trừ hết điểm", "kiểm tra lý thuyết", "mô phỏng"]
    ):
        score += 35
    if "chưa bị trừ hết điểm" in q and _has_any(
        t, ["chưa bị trừ hết điểm", "12 tháng", "phục hồi đủ 12 điểm"]
    ):
        score += 45
    if _has_any(q, ["biển báo", "báo hiệu", "cọc tiêu"]) and _has_any(
        t, ["báo hiệu đường bộ", "biển báo", "cọc tiêu", "vạch kẻ"]
    ):
        score += 30
    if _has_any(q, ["đăng ký xe", "biển số xe"]) and _has_any(
        t, ["đăng ký xe", "biển số xe", "hồ sơ đăng ký"]
    ):
        score += 30

    return score


def _token_overlap_score(question: str, text: str) -> int:
    q_tokens = Counter(_tokenize(question))
    t_tokens = Counter(_tokenize(text))
    return sum(min(count, t_tokens[token]) for token, count in q_tokens.items())


def _article_bonus(text: str) -> int:
    return 5 if _normalize(text).startswith("điều ") else 0


def _article_intent_bonus(question: str, article: str, doc_id: str) -> int:
    bonus = 0
    article = article or ""
    intents = classify_intents(question)

    for intent in intents:
        if intent == "passenger" and "passenger" not in classify_intents(question):
            continue
        for rank, hint in enumerate(ARTICLE_HINTS_BY_INTENT.get(intent, [])):
            if hint in article:
                bonus += 30 - rank * 5

    if "restore_points" in intents and doc_id == "tt_65_2024_tt_bca":
        bonus += 40
    if "restore_points_partial" in intents and doc_id == "nd_168_2024_nd_cp":
        bonus += 45
    if "passenger" in intents and "Điều 12." in article:
        bonus += 25
    if "helmet" in intents and "Điều 6." in article:
        bonus += 20

    return bonus


def _legal_completeness_bonus(question: str, text: str) -> int:
    q = _normalize(question)
    t = _normalize(text)
    bonus = 0

    asks_penalty = _has_any(q, ["phạt", "xử phạt", "bao nhiêu", "mức nào"])
    asks_points = _has_any(q, ["trừ điểm", "giấy phép lái xe", "gplx"])
    asks_restore = _has_any(q, ["phục hồi", "trừ hết điểm", "kiểm tra"])

    if asks_penalty and "phạt tiền từ" in t:
        bonus += 45
    if asks_points and "trừ điểm giấy phép lái xe" in t:
        bonus += 40
    if asks_restore and _has_any(t, ["phục hồi điểm", "phục hồi đủ 12 điểm",
                                      "kiểm tra lý thuyết", "mô phỏng"]):
        bonus += 40
    if "80" in q and _has_any(t, ["vượt quá 80 miligam", "vượt quá 0,4 miligam",
                                   "vượt quá 0.4 miligam"]):
        bonus += 30

    return bonus


def _subject_role_bonus(question: str, text: str) -> int:
    q = _normalize(question)
    t = _normalize(text)
    bonus = 0

    if _has_any(q, ["ô tô", "oto", "xe hơi"]):
        if _has_any(t, ["xe ô tô", "các loại xe tương tự xe ô tô",
                         "xe chở người bốn bánh", "xe chở hàng bốn bánh",
                         "người điều khiển ô tô"]):
            bonus += 35
        if _has_any(t, ["xe đạp", "xích lô", "xe đạp máy", "xe thô sơ"]):
            bonus -= 35

    if _has_any(q, ["xe máy", "mô tô", "gắn máy"]):
        if _has_any(t, ["người điều khiển xe", "khi điều khiển xe",
                         "xe mô tô", "xe gắn máy"]):
            bonus += 25
        if _has_any(t, ["xe đạp", "xích lô", "xe đạp máy", "xe thô sơ"]):
            bonus -= 25

    if _has_any(q, ["người được chở", "người ngồi sau", "ngồi sau", "hành khách"]):
        if _has_any(t, ["người được chở", "người ngồi trên xe"]):
            bonus += 30
    elif "mũ bảo hiểm" in q and "người được chở" in t:
        bonus -= 15

    return bonus


def _lexical_score(question: str, doc) -> int:
    """Combined lexical scoring (token overlap + phrase + legal + subject + article)."""
    md = doc.metadata or {}
    doc_id = md.get("doc_id") or md.get("source") or ""
    article = md.get("article") or ""
    text = doc.page_content or ""
    score = 0
    expanded = expand_traffic_query(question)
    score += _doc_priority(question, doc_id)
    score += _phrase_score(question, text)
    score += _token_overlap_score(expanded, text)
    score += _article_bonus(text)
    score += _article_intent_bonus(question, article, doc_id)
    score += _legal_completeness_bonus(question, text)
    score += _subject_role_bonus(question, text)
    return score


# ── Cross-encoder reranker ─────────────────────────────────────────────────────

_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cross_encoder = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).eval().to(device),
            AutoTokenizer.from_pretrained(model_name),
            device,
        )
    return _cross_encoder


def _cross_encoder_scores(question: str, docs: list) -> list[float]:
    """Score (question, doc) pairs with cross-encoder, return scores per doc."""
    if not docs:
        return []

    import torch
    model, tokenizer, device = _get_cross_encoder()
    pairs = [(question, doc.page_content[:2048]) for doc in docs]

    scores = []
    batch_size = 8
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        with torch.inference_mode():
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=2048, return_tensors="pt",
            ).to(device)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())

    return scores


# ── Core retrieval pipeline ────────────────────────────────────────────────────

def _vectorstore_docs(vectorstore) -> list:
    docstore = getattr(vectorstore, "docstore", None)
    docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    return list(docs.values())


def _doc_signature(doc) -> tuple:
    md = doc.metadata or {}
    chunk_id = md.get("chunk_id")
    if chunk_id:
        return ("chunk", chunk_id)
    return (md.get("doc_id") or md.get("source") or "",
            _normalize(doc.page_content or "")[:300])


def _append_unique(target: list, docs: list, seen: set) -> None:
    for doc in docs:
        key = _doc_signature(doc)
        if key not in seen:
            target.append(doc)
            seen.add(key)


def _same_article(a: dict, b: dict) -> bool:
    return (
        (a.get("doc_id") or a.get("source") or "")
        == (b.get("doc_id") or b.get("source") or "")
        and (a.get("article") or "")
        == (b.get("article") or "")
    )


def _point_supplement_for(primary, docs: list):
    md = primary.metadata or {}
    clause_number = md.get("clause_number")
    text_norm = _normalize(primary.page_content or "")
    if not clause_number or "phạt tiền từ" not in text_norm:
        return None

    clause_ref = f"khoản {clause_number}"
    for doc in docs:
        if _doc_signature(doc) == _doc_signature(primary):
            continue
        doc_md = doc.metadata or {}
        doc_text = _normalize(doc.page_content or "")
        if not _same_article(md, doc_md):
            continue
        if "bị trừ điểm giấy phép lái xe như sau" not in doc_text:
            continue
        if clause_ref in doc_text:
            return doc
    return None


def _with_structural_supplements(ranked: list, top_k: int,
                                  all_docs: list | None = None) -> list:
    if not ranked:
        return []
    result = ranked[:top_k]
    seen = {_doc_signature(doc) for doc in result}
    supplement = _point_supplement_for(ranked[0], ranked)
    if supplement is None and all_docs is not None:
        supplement = _point_supplement_for(ranked[0], all_docs)
    if supplement is None or _doc_signature(supplement) in seen:
        return result
    rest = [doc for doc in result[1:] if _doc_signature(doc) != _doc_signature(supplement)]
    return [result[0], supplement, *rest][:top_k]


def retrieve_ranked_docs(vectorstore, question: str, top_k: int = 3,
                          candidate_k: int = 30, min_score: float | None = None) -> list:
    """FAISS semantic → lexical re-rank → cross-encoder re-rank → structural supplements.
    
    Supports multi-hop retrieval: after initial FAISS pass, if the top result has
    low relevance, extract article references and re-query for better coverage.
    
    Args:
        vectorstore: FAISS vector store
        question: Vietnamese question string
        top_k: number of documents to return
        candidate_k: FAISS candidate pool size (larger = better recall)
        min_score: if not None, return empty list when best cross-encoder score < threshold

    Returns:
        Ordered list of Document objects
    """
    if not is_supported_traffic_question(question):
        return []

    expanded = expand_traffic_query(question)
    candidates = []
    seen = set()

    # ── Step 1: FAISS semantic search (always primary) ──
    try:
        semantic = vectorstore.similarity_search(expanded, k=candidate_k)
        _append_unique(candidates, semantic, seen)
    except Exception:
        pass

    # ── Step 1b: Multi-hop — extract article from question and re-query ──
    article_match = re.search(r'[Đđ]iều\s+(\d+)', question)
    if article_match:
        article_num = article_match.group(1)
        article_query = f"{expanded} điều {article_num}"
        try:
            more_docs = vectorstore.similarity_search(article_query, k=candidate_k // 2)
            matching = []
            for doc in more_docs:
                md = doc.metadata or {}
                art = (md.get("article") or "").lower()
                if f"điều {article_num}" in art:
                    matching.append(doc)
            _append_unique(candidates, matching, seen)
        except Exception:
            pass

        direct_matches = []
        preferred_docs = {
            doc_id
            for intent in classify_intents(question)
            for doc_id in DOC_PRIORITY_BY_INTENT.get(intent, [])
        }
        for doc in _vectorstore_docs(vectorstore):
            md = doc.metadata or {}
            if str(md.get("article_number") or "") != article_num:
                continue
            if preferred_docs and (md.get("doc_id") or md.get("source") or "") not in preferred_docs:
                continue
            direct_matches.append(doc)
        _append_unique(candidates, direct_matches, seen)

    # ── Step 2: Lexical candidates as fallback/supplement ──
    if len(candidates) < candidate_k:
        lexical_scored = []
        for doc in _vectorstore_docs(vectorstore):
            key = _doc_signature(doc)
            if key in seen:
                continue
            score = _lexical_score(question, doc)
            if score > 0:
                lexical_scored.append((score, doc))
        lexical_scored.sort(key=lambda x: x[0], reverse=True)
        gap = candidate_k - len(candidates)
        for _, doc in lexical_scored[:gap]:
            key = _doc_signature(doc)
            if key not in seen:
                candidates.append(doc)
                seen.add(key)

    if not candidates:
        return []

    # ── Step 3: Lexical re-ranking ──
    scored_candidates = []
    for idx, doc in enumerate(candidates):
        score = _lexical_score(question, doc) - idx * 0.1
        scored_candidates.append((score, doc))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # ── Step 4: Cross-encoder re-rank top 20 ──
    rerank_pool = [doc for _, doc in scored_candidates[:min(20, len(scored_candidates))]]
    ce_scores = []
    if len(rerank_pool) >= 2:
        try:
            ce_scores = _cross_encoder_scores(question, rerank_pool)
            scored_candidates = sorted(
                zip(ce_scores, rerank_pool), key=lambda x: x[0], reverse=True
            )
            reranked = [doc for _, doc in scored_candidates]
            for doc in candidates:
                if _doc_signature(doc) not in {_doc_signature(d) for d in reranked}:
                    reranked.append(doc)
            ranked_docs = reranked
        except Exception:
            ranked_docs = [doc for _, doc in scored_candidates]
    else:
        ranked_docs = [doc for _, doc in scored_candidates]

    # ── Step 4b: Cross-encoder score gate ──
    if min_score is not None and ce_scores:
        if max(ce_scores) < min_score:
            return []

    return _with_structural_supplements(ranked_docs, top_k, all_docs=_vectorstore_docs(vectorstore))
