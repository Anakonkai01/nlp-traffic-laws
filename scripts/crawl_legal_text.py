"""
Crawl text versions of scanned Vietnamese legal PDFs.

Tries sources in order: thuvienphapluat.vn → vbpl.vn → luatvietnam.vn
Output: docs/docs_giaothong/text/<slug>.txt

Usage:
    python scripts/crawl_legal_text.py          # crawl all configured docs
    python scripts/crawl_legal_text.py --dry-run # show what would be fetched
"""

import argparse
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "docs_giaothong" / "text"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# (slug, search_keyword, description)
DOCUMENTS = [
    ("nd_168_2024_nd_cp",  "168/2024/NĐ-CP",  "NĐ 168/2024/NĐ-CP — xử phạt vi phạm giao thông đường bộ"),
    ("nd_158_2024_nd_cp",  "158/2024/NĐ-CP",  "NĐ 158/2024/NĐ-CP — quy tắc giao thông đường bộ"),
    ("nd_165_2024_nd_cp",  "165/2024/NĐ-CP",  "NĐ 165/2024/NĐ-CP"),
    ("nd_336_2025_nd_cp",  "336/2025/NĐ-CP",  "NĐ 336/2025/NĐ-CP"),
    ("tt_65_2024_tt_bca",  "65/2024/TT-BCA",  "TT 65/2024/TT-BCA — Bộ Công an"),
    ("qcvn_41_2024_bgtvt", "QCVN 41:2024",    "QCVN 41:2024/BGTVT — báo hiệu đường bộ"),
    ("tt_05_2025_tt_bgtvt","05/2025/TT-BGTVT","TT 05/2025/TT-BGTVT — đào tạo sát hạch lái xe"),
    ("nd_39_2025_nd_cp",   "39/2025/NĐ-CP",   "NĐ 39/2025/NĐ-CP — đấu giá biển số xe ô tô"),
    ("tt_79_2024_tt_bca",  "79/2024/TT-BCA",  "TT 79/2024/TT-BCA — đăng ký xe, biển số xe"),
]


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[̀-ͯ]", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Collapse excessive whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_main_text(soup: BeautifulSoup) -> str:
    """Try common content selectors, return largest non-empty match."""
    selectors = [
        "#toanvan", ".content1", "#noidung", ".full-content",
        "article.content", "div.content", "#vanban", ".vanban-content",
        "#divNoidung", ".box-content", "div[id*='content']",
    ]
    candidates = []
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(separator="\n", strip=True)
            if len(t) > 300:
                candidates.append(t)

    if candidates:
        return max(candidates, key=len)

    # Fallback: largest <div>
    divs = [d.get_text(separator="\n", strip=True) for d in soup.find_all("div")]
    divs = [d for d in divs if len(d) > 500]
    return max(divs, key=len) if divs else ""


# ---------------------------------------------------------------------------
# Source: thuvienphapluat.vn
# ---------------------------------------------------------------------------

def try_thuvienphapluat(keyword: str) -> str:
    """Search thuvienphapluat.vn and return text of first matching doc."""
    # Their search endpoint accepts keyword as query param
    url = "https://thuvienphapluat.vn/van-ban/Tim-Van-Ban.aspx"
    try:
        resp = SESSION.get(url, params={"keyword": keyword, "aHref": "timvb"}, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find first document link in results
        doc_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/van-ban/" in href and ".aspx" in href and "Tim-Van-Ban" not in href:
                doc_url = href if href.startswith("http") else "https://thuvienphapluat.vn" + href
                break

        if not doc_url:
            return ""

        time.sleep(1)
        resp2 = SESSION.get(doc_url, timeout=30)
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        text = extract_main_text(soup2)
        return clean_text(text) if len(text) > 500 else ""

    except requests.RequestException:
        return ""


# ---------------------------------------------------------------------------
# Source: vbpl.vn (official gov database)
# ---------------------------------------------------------------------------

def try_vbpl(keyword: str) -> str:
    """Search vbpl.vn for the document and return its text."""
    # AJAX search endpoint
    search_url = "https://vbpl.vn/TW/Pages/vbpq-timkiem.aspx"
    try:
        resp = SESSION.get(search_url, params={"type": "0", "s": keyword}, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        doc_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "vbpq-van-ban-phap-luat" in href or "vbpq-chi-tiet" in href:
                doc_url = href if href.startswith("http") else "https://vbpl.vn" + href
                break

        if not doc_url:
            return ""

        time.sleep(1)
        resp2 = SESSION.get(doc_url, timeout=30)
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        text = extract_main_text(soup2)
        return clean_text(text) if len(text) > 500 else ""

    except requests.RequestException:
        return ""


# ---------------------------------------------------------------------------
# Source: luatvietnam.vn
# ---------------------------------------------------------------------------

def try_luatvietnam(keyword: str) -> str:
    """Search luatvietnam.vn for the document."""
    search_url = "https://luatvietnam.vn/tim-kiem/"
    try:
        resp = SESSION.get(search_url, params={"q": keyword}, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        doc_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "luatvietnam.vn" in href and keyword.split("/")[0] in href:
                doc_url = href
                break
            elif href.startswith("/") and keyword.split("/")[0] in href:
                doc_url = "https://luatvietnam.vn" + href
                break

        if not doc_url:
            return ""

        time.sleep(1)
        resp2 = SESSION.get(doc_url, timeout=30)
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        text = extract_main_text(soup2)
        return clean_text(text) if len(text) > 500 else ""

    except requests.RequestException:
        return ""


# ---------------------------------------------------------------------------
# Main crawl logic
# ---------------------------------------------------------------------------

def crawl_one(slug: str, keyword: str, description: str, dry_run: bool = False) -> bool:
    out_path = OUTPUT_DIR / f"{slug}.txt"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  [{slug}] Already exists ({out_path.stat().st_size:,} B) — skip.")
        return True

    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Keyword: {keyword}")
    if dry_run:
        print("  [dry-run] Would crawl from thuvienphapluat → vbpl → luatvietnam")
        return False

    for source_name, fn in [
        ("thuvienphapluat.vn", try_thuvienphapluat),
        ("vbpl.vn",            try_vbpl),
        ("luatvietnam.vn",     try_luatvietnam),
    ]:
        print(f"  Trying {source_name}...", end=" ", flush=True)
        text = fn(keyword)
        if text:
            out_path.write_text(text, encoding="utf-8")
            print(f"OK — {len(text):,} chars → {out_path.name}")
            return True
        print("not found")
        time.sleep(1)

    print(f"  FAILED — all sources exhausted for {keyword}")
    print(f"  → Please download manually from thuvienphapluat.vn and save to:")
    print(f"    {out_path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, total = 0, len(DOCUMENTS)
    for slug, keyword, description in DOCUMENTS:
        if crawl_one(slug, keyword, description, dry_run=args.dry_run):
            ok += 1
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Result: {ok}/{total} documents obtained.")
    if ok < total:
        print("For missing documents, download the text version manually from thuvienphapluat.vn")
        print(f"and place .txt files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
