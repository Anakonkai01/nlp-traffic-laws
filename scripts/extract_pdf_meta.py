"""Extract metadata from scanned PDFs to determine document numbers."""
import fitz
import os

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "docs_giaothong")

scanned = [
    "158_2024_nd-cp_18122024-signed.pdf",
    "165-pl1.pdf",
    "165_2024_nd-cp_26122024-signed.pdf",
    "168-nd-cp.signed.pdf",
    "336nd.signed.pdf",
    "65-bca.signed.pdf",
]

for fname in scanned:
    path = os.path.join(DOCS_DIR, fname)
    doc = fitz.open(path)
    meta = doc.metadata
    print(f"=== {fname} ({len(doc)}p) ===")
    for k, v in meta.items():
        if v:
            print(f"  {k}: {v}")
    # Check first 2 pages for doc number in header text
    for i, page in enumerate(doc[:2]):
        words = page.get_text("words")
        if words:
            line = " ".join(w[4] for w in words[:40])
            print(f"  page{i}_text: {line[:200]}")
    print()
