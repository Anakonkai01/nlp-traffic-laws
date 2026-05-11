"""Traffic-law question classification and query expansion utilities.

These are used by evaluate.py for retrieval diagnostics (intent tags, expanded
query), and by sanction_facts.py for violation-term expansion during fact BM25.
They are intentionally kept here rather than in the retrieval path so they
survive the v2 removal.
"""

from text_utils import normalize


DOC_ID_ALIASES = {
    "nd_168_2024_nd_cp": ["nghị định 168", "168/2024", "168-2024", "168/2024/nđ-cp", "168/2024/nd-cp"],
    "nd_158_2024_nd_cp": ["nghị định 158", "158/2024", "158-2024"],
    "nd_165_2024_nd_cp": ["nghị định 165", "165/2024", "165-2024"],
    "nd_336_2025_nd_cp": ["nghị định 336", "336/2025", "336-2025"],
    "tt_65_2024_tt_bca": ["thông tư 65", "65/2024", "65-2024"],
    "tt_79_2024_tt_bca": ["thông tư 79", "79/2024", "79-2024"],
    "tt_05_2025_tt_bgtvt": ["thông tư 05", "thông tư 12", "05/2025", "12/2025"],
    "luat_35_2024_qh15": ["luật 35", "35/2024", "35-2024", "luật đường bộ"],
    "luat_36_2024_qh15": ["luật 36", "36/2024", "36-2024", "luật trật tự"],
    "qcvn_41_2024_bgtvt": ["qcvn 41", "41:2024", "quy chuẩn 41"],
    "nd_39_2023_nd_cp": ["nghị định 39", "39/2023", "đấu giá biển số"],
    "tt_suckhoe_lai_xe": ["sức khỏe lái xe", "khám sức khỏe", "tiêu chuẩn sức khỏe"],
}

INTENT_PHRASES = {
    "signal":     "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "helmet":     "không đội mũ bảo hiểm cho người đi mô tô xe máy",
    "alcohol":    "nồng độ cồn trong máu hoặc hơi thở",
    "license_points": "trừ điểm giấy phép lái xe phục hồi điểm giấy phép lái xe",
    "health": "khám sức khỏe tiêu chuẩn sức khỏe người lái xe",
    "transport_operation": "kinh doanh vận tải phù hiệu hợp đồng vận tải",
    "international_transport": "giấy phép liên vận GMS sổ TAD Lào Campuchia phương tiện phi thương mại",
    "road_infrastructure": "quản lý bảo vệ kết cấu hạ tầng đường bộ quốc lộ cọc tiêu hộ lan dự án PPP",
    "road_safety_audit": "thẩm tra an toàn giao thông thẩm định an toàn giao thông thẩm tra viên chủ nhiệm thẩm tra",
    "admin_sanction": "thẩm quyền xử phạt biện pháp khắc phục hậu quả xử phạt bổ sung cảnh sát cơ động",
    "general_traffic_law": "quy tắc giao thông đường bộ người tham gia giao thông đường bộ",
}

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
    "liên vận":       "giấy phép liên vận GMS sổ TAD Lào Campuchia phương tiện phi thương mại",
    "gms":            "giấy phép liên vận GMS vận tải đường bộ quốc tế",
    "sổ tad":         "sổ TAD giấy phép liên vận phương tiện vận tải quốc tế",
    "lào":            "phương tiện của Lào Campuchia gia hạn lưu hành tại Việt Nam",
    "campuchia":      "phương tiện của Lào Campuchia gia hạn lưu hành tại Việt Nam",
    "ppp":            "dự án PPP đường bộ báo cáo nghiên cứu tiền khả thi chủ trương đầu tư hợp đồng dự án",
    "dự án ppp":      "dự án PPP đường bộ báo cáo nghiên cứu tiền khả thi chủ trương đầu tư hợp đồng dự án",
    "quốc lộ":        "quản lý quốc lộ kết cấu hạ tầng đường bộ Bộ Giao thông vận tải Ủy ban nhân dân cấp tỉnh",
    "cọc tiêu":       "cọc tiêu hộ lan công trình an toàn đường bộ kết cấu hạ tầng đường bộ",
    "hộ lan":         "hộ lan cọc tiêu công trình an toàn đường bộ kết cấu hạ tầng đường bộ",
    "thẩm tra an toàn giao thông": "thẩm tra thẩm định an toàn giao thông đường bộ thẩm tra viên chủ nhiệm thẩm tra",
    "thẩm định an toàn giao thông": "thẩm tra thẩm định an toàn giao thông đường bộ thẩm tra viên chủ nhiệm thẩm tra",
    "cảnh sát cơ động": "cảnh sát cơ động thẩm quyền xử phạt vi phạm hành chính giao thông đường bộ",
    "thẩm quyền xử phạt": "thẩm quyền xử phạt vi phạm hành chính biện pháp khắc phục hậu quả xử phạt bổ sung",
    "xử phạt bổ sung": "hình thức xử phạt bổ sung biện pháp khắc phục hậu quả vi phạm hành chính",
    "biện pháp khắc phục": "biện pháp khắc phục hậu quả vi phạm hành chính buộc thực hiện",
}


def _has_any(text: str, phrases: list[str]) -> bool:
    lowered = normalize(text)
    return any(normalize(phrase) in lowered for phrase in phrases)


def classify_intents(question: str) -> list[str]:
    q = normalize(question)
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
    if _has_any(q, ["hiệu lực", "khi nào có hiệu lực", "ngày hiệu lực", "thời điểm có hiệu lực"]):
        intents.append("effective_date")
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
    if _has_any(q, ["liên vận", "gms", "sổ tad", "lào", "campuchia", "phi thương mại"]):
        intents.append("international_transport")
    if _has_any(q, ["ppp", "dự án", "quốc lộ", "kết cấu hạ tầng", "cọc tiêu", "hộ lan"]):
        intents.append("road_infrastructure")
    if _has_any(q, ["thẩm tra an toàn giao thông", "thẩm định an toàn giao thông", "thẩm tra viên", "chủ nhiệm thẩm tra"]):
        intents.append("road_safety_audit")
    if _has_any(q, ["thẩm quyền xử phạt", "cảnh sát cơ động", "xử phạt bổ sung", "biện pháp khắc phục", "hậu quả",
                    "vỉa hè", "lòng đường", "hành lang an toàn", "chiếm dụng đất", "thi công", "gửi thông báo",
                    "đập phá", "tháo dỡ", "bó vỉa", "điều 39", "điều 40", "điều 43"]):
        intents.append("admin_sanction")
    if _has_any(q, ["quy tắc giao thông", "người tham gia giao thông", "luật đường bộ", "kết cấu hạ tầng",
                    "đường cao tốc", "đại lý thu gom", "kho bãi hàng"]):
        intents.append("general_traffic_law")

    return list(dict.fromkeys(intents))


def expand_traffic_query(question: str) -> str:
    q = normalize(question)
    intents = classify_intents(question)
    expansions: list[str] = []

    for intent in (
        "signal", "helmet", "alcohol", "license_points", "health",
        "transport_operation", "international_transport", "road_infrastructure",
        "road_safety_audit", "admin_sanction", "general_traffic_law",
    ):
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
    if "international_transport" in intents:
        expansions.append("giấy phép liên vận GMS sổ TAD Lào Campuchia phương tiện phi thương mại")
    if "road_infrastructure" in intents:
        expansions.append("quản lý bảo vệ kết cấu hạ tầng đường bộ quốc lộ cọc tiêu hộ lan dự án PPP")
    if "road_safety_audit" in intents:
        expansions.append("thẩm tra an toàn giao thông thẩm định an toàn giao thông thẩm tra viên")
    if "admin_sanction" in intents:
        expansions.append("thẩm quyền xử phạt vi phạm hành chính biện pháp khắc phục hậu quả xử phạt bổ sung")
    if "general_traffic_law" in intents:
        expansions.append("quy tắc giao thông đường bộ người tham gia giao thông đường bộ")

    for surface, legal in QUERY_TO_LEGAL.items():
        if normalize(surface) in q and legal not in "\n".join(expansions):
            expansions.append(legal)

    if not expansions:
        return question
    return question + "\n" + "\n".join(dict.fromkeys(expansions))
