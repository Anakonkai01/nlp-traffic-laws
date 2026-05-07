import os
from pathlib import Path

# Chunking must stay consistent across KB build, QA generation, and eval.
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 2000
SEPARATORS = ["Điều ", "Khoản ", "Điểm ", "\n\n", "\n", " "]

PROJECT_ROOT = Path(__file__).parent.parent

DOMAIN_SLUG = "traffic"
KB_SOURCE_POLICY = "local_text_only"

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
TRAFFIC_DOCS_DIR = DOCS_DIR / "docs_giaothong"
TRAFFIC_TEXT_DIR = TRAFFIC_DOCS_DIR / "text"
SOURCE_MANIFEST_PATH = TRAFFIC_DOCS_DIR / "manifest.json"

QA_SPLIT_DIR = Path(os.environ.get("QA_SPLIT_DIR", DATA_DIR / "splits_filtered"))
if not QA_SPLIT_DIR.is_absolute():
    QA_SPLIT_DIR = PROJECT_ROOT / QA_SPLIT_DIR
SPLITS_FILTERED_DIR = QA_SPLIT_DIR
QA_DATA_PATH = QA_SPLIT_DIR / "qa_train.jsonl"
QA_DEV_PATH = QA_SPLIT_DIR / "qa_dev.jsonl"
QA_TEST_PATH = QA_SPLIT_DIR / "qa_test.jsonl"
QA_LEGACY_DATA_PATH = DATA_DIR / "qa_pairs_traffic.jsonl"
QA_V2_DATA_PATH = DATA_DIR / "qa_pairs_traffic_v2.jsonl"
EVAL_DATA_PATH = DATA_DIR / "eval_manual.jsonl"
EVAL_MC_DATA_PATH = DATA_DIR / "eval_mc_manual.jsonl"
KB_PATH = PROJECT_ROOT / "vector_db_traffic"
KB_META_PATH = KB_PATH / "build_meta.json"

MODEL_ID = "Qwen/Qwen3.5-9B"
OLLAMA_MODEL = "qwen3.5:9b"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3.5-9b-lora-traffic-v2"
MODEL_DIR_V2 = MODEL_DIR  # backward-compatible alias
RERANKER_FT_DIR = PROJECT_ROOT / "models" / "bge-reranker-v2-m3-traffic-ft"
REPORTS_DIR = PROJECT_ROOT / "reports" / DOMAIN_SLUG

TRAFFIC_TITLE_KEYWORDS = [
    "giao thông đường bộ",
    "trật tự, an toàn giao thông đường bộ",
    "luật đường bộ",
    "luật giao thông",
    "giấy phép lái xe",
    "đăng ký xe",
    "đăng kiểm xe",
    "kiểm định xe",
    "biển báo hiệu đường bộ",
    "báo hiệu đường bộ",
    "xử phạt vi phạm giao thông",
    "vi phạm hành chính trong lĩnh vực giao thông",
    "vận tải đường bộ",
    "an toàn giao thông",
    "nồng độ cồn",
    "phương tiện giao thông đường bộ",
]

TRAFFIC_CONTENT_KEYWORDS = [
    "giao thông đường bộ",
    "trật tự, an toàn giao thông đường bộ",
    "luật đường bộ",
    "giấy phép lái xe",
    "điểm của giấy phép lái xe",
    "đăng ký xe",
    "đăng kiểm xe",
    "kiểm định xe cơ giới",
    "biển báo hiệu đường bộ",
    "báo hiệu đường bộ",
    "vạch kẻ đường",
    "tín hiệu đèn giao thông",
    "nồng độ cồn",
    "xử phạt vi phạm giao thông",
    "người tham gia giao thông đường bộ",
    "phương tiện giao thông đường bộ",
    "vận tải đường bộ",
    "xe cơ giới",
    "xe máy chuyên dùng",
    "trật tự an toàn giao thông",
]

TRAFFIC_MC_KEYWORDS = sorted(set(TRAFFIC_TITLE_KEYWORDS + TRAFFIC_CONTENT_KEYWORDS + [
    "đèn giao thông",
    "đèn đỏ",
    "đèn vàng",
    "biển số xe",
    "biển báo",
    "làn đường",
    "phần đường",
    "đường cao tốc",
    "đỗ xe",
    "dừng xe",
    "vượt xe",
    "điều khiển xe",
    "người lái xe",
    "xe ô tô",
    "xe máy",
    "mũ bảo hiểm",
    "chuyển hướng",
    "tốc độ",
    "nút giao",
    "xe ưu tiên",
]))

REFUSAL_ANSWER = (
    "Tôi không tìm thấy căn cứ trong các văn bản giao thông đã được cung cấp "
    "để trả lời câu hỏi này."
)

FORBIDDEN_LEGACY_TERMS = [
    "Luật Giao thông đường bộ 2008",
    "23/2008/QH12",
    "Nghị định 100/2019",
    "100/2019/NĐ-CP",
    "Nghị định 46/2016",
    "46/2016/NĐ-CP",
    "Nghị định 123/2021",
    "123/2021/NĐ-CP",
    "dự thảo",
]

# For configs A/C (no RAG): answer from general knowledge
TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT = (
    "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. "
    "Hãy trả lời câu hỏi dựa trên kiến thức của bạn về pháp luật giao thông Việt Nam hiện hành (2024-2025). "
    "Trả lời ngắn gọn, chính xác, bằng tiếng Việt và nêu căn cứ văn bản/điều khoản nếu biết."
)

# For configs B/D (RAG): prioritize context, fall back to general knowledge
TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT = (
    "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. "
    "Đoạn văn bản luật có thể gồm các nguồn riêng biệt; chỉ dùng nội dung nằm trong cùng một nguồn/căn cứ, không chép danh sách điểm từ nguồn khác. "
    "Trả lời trực tiếp kết quả cuối cùng, không trích nguyên văn đoạn dài. "
    "Trả lời chủ yếu dựa trên đoạn văn bản luật được cung cấp và nêu rõ căn cứ văn bản/điều khoản nếu có. "
    "Nếu đoạn trích có cùng hành vi, phương tiện, mức phạt, thủ tục hoặc cùng điều luật liên quan, hãy trả lời phần có căn cứ thay vì từ chối. "
    "Nếu câu hỏi thuộc lĩnh vực giao thông nhưng đoạn trích chưa đủ chi tiết, hãy trả lời ngắn gọn từ kiến thức pháp luật giao thông hiện hành và ghi rõ 'Cần kiểm tra lại căn cứ'. "
    "Chỉ từ chối khi câu hỏi không thuộc lĩnh vực giao thông hoặc hoàn toàn không liên quan đến các văn bản được cung cấp; khi đó hãy trả lời đúng câu sau: "
    f"'{REFUSAL_ANSWER}' "
    "Không viện dẫn văn bản cũ đã hết hiệu lực. "
    "Khi nêu căn cứ, hãy ghi đầy đủ Điều, khoản và điểm nếu có (định dạng 'Điều X khoản Y điểm Z') "
    "và nêu cụ thể số tiền, số điểm trừ hoặc thời hạn tước giấy phép nếu có trong đoạn trích. "
    "Trả lời ngắn gọn, chính xác, bằng tiếng Việt."
)

# Backward-compat alias (used by finetune.py for the no-context dropout branch)
TRAFFIC_QA_SYSTEM_PROMPT = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT

TRAFFIC_MC_SYSTEM_PROMPT_NO_CONTEXT = (
    "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. "
    "Dựa trên kiến thức của bạn, hãy chọn đáp án đúng nhất cho câu hỏi trắc nghiệm. "
    "Chỉ trả lời bằng MỘT chữ cái duy nhất: A, B, C hoặc D."
)

TRAFFIC_MC_SYSTEM_PROMPT_WITH_CONTEXT = (
    "Bạn là chuyên gia pháp luật giao thông đường bộ Việt Nam. "
    "Dựa trên đoạn văn bản luật được cung cấp, hãy chọn đáp án đúng nhất cho câu hỏi trắc nghiệm. "
    "Chỉ trả lời bằng MỘT chữ cái duy nhất: A, B, C hoặc D."
)

# Backward-compat alias
TRAFFIC_MC_SYSTEM_PROMPT = TRAFFIC_MC_SYSTEM_PROMPT_NO_CONTEXT
