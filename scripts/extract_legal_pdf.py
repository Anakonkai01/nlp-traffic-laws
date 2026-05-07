"""
Extract full legal text from Vietnamese law PDFs.

Output:
  docs/docs_giaothong/text/luat_35_2024_qh15.txt  <- Luật Đường bộ
  docs/docs_giaothong/text/luat_36_2024_qh15.txt  <- Luật Trật tự, an toàn GTĐB
"""
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

DOCS_DIR = Path(__file__).parent.parent / "docs" / "docs_giaothong"
OUT_DIR = DOCS_DIR / "text"

# Header pattern in Công báo PDFs: "CÔNG BÁO/Số NNN + NNN/Ngày DD-M-YYYY"
_CONGBAO_RE = re.compile(r"CÔNG BÁO/Số\s+\d+\s*\+\s*\d+/Ngày\s+\d+-\d+-\d+")
# Isolated page number line (1-4 digits, optionally surrounded by spaces)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
# Repeated preamble in continuation PDFs
_CONT_PREAMBLE_RE = re.compile(
    r"VĂN BẢN QUY PHẠM PHÁP LUẬT.*?Tiếp theo[^\n]*\n",
    re.DOTALL,
)
# Cross-reference lines like "(Xem tiếp Công báo số 979 + 980)"
_XEMTIEP_RE = re.compile(r"\(Xem tiếp Công báo[^\)]*\)")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_page(text: str) -> str:
    """Remove Công báo header, isolated page numbers, and normalize whitespace."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if _CONGBAO_RE.search(line):
            continue
        if _PAGE_NUM_RE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        raw = page.get_text("text")
        pages.append(_clean_page(raw))
    return "\n".join(pages)


def _remove_continuation_preamble(text: str) -> str:
    """Strip the index/title block at the start of a '_tiep' continuation PDF."""
    # Find the first Điều marker and take from there, keeping a small buffer
    match = re.search(r"\nĐiều\s+\d+", text)
    if match:
        return text[match.start():].lstrip("\n")
    return text


def _post_process(text: str) -> str:
    """Normalize NFC, collapse excessive blank lines, strip trailing spaces."""
    text = _nfc(text)
    text = _XEMTIEP_RE.sub("", text)
    # Remove duplicate blank lines (keep max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def build_luat_35() -> Path:
    """Luật Đường bộ — single PDF 35-2024-qh15.pdf."""
    src = DOCS_DIR / "35-2024-qh15.pdf"
    out = OUT_DIR / "luat_35_2024_qh15.txt"
    print(f"Extracting {src.name} ...", end=" ", flush=True)
    text = extract_pdf_text(src)
    text = _post_process(text)
    out.write_text(text, encoding="utf-8")
    print(f"done → {len(text):,} chars, {text.count(chr(10)):,} lines")
    return out


def build_luat_36() -> Path:
    """Luật Trật tự, an toàn GTĐB — 36-2024-qh15.pdf + 36-2024-qh15_tiep.pdf."""
    src1 = DOCS_DIR / "36-2024-qh15.pdf"
    src2 = DOCS_DIR / "36-2024-qh15_tiep.pdf"
    out = OUT_DIR / "luat_36_2024_qh15.txt"

    print(f"Extracting {src1.name} ...", end=" ", flush=True)
    part1 = extract_pdf_text(src1)
    print(f"{len(part1):,} chars")

    print(f"Extracting {src2.name} ...", end=" ", flush=True)
    part2_raw = extract_pdf_text(src2)
    part2 = _remove_continuation_preamble(part2_raw)
    print(f"{len(part2):,} chars (after stripping preamble)")

    combined = part1.rstrip() + "\n\n" + part2.lstrip()
    combined = _post_process(combined)
    out.write_text(combined, encoding="utf-8")
    print(f"Combined → {len(combined):,} chars, {combined.count(chr(10)):,} lines → {out}")
    return out


def verify(path: Path) -> None:
    """Print first Điều titles found to confirm extraction quality."""
    text = path.read_text(encoding="utf-8")
    articles = re.findall(r"Điều\s+\d+[a-zA-Z]?\.[^\n]{0,80}", text)
    print(f"\n{path.name}: {len(articles)} Điều found, first 5:")
    for a in articles[:5]:
        print(f"  {a.strip()}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out35 = build_luat_35()
    out36 = build_luat_36()
    verify(out35)
    verify(out36)
    print("\nDone. Add these to manifest.json to include in the KB.")
