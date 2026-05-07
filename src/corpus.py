import json
import random
import re
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from config import (
    KB_SOURCE_POLICY,
    PROJECT_ROOT,
    SOURCE_MANIFEST_PATH,
    TRAFFIC_CONTENT_KEYWORDS,
    TRAFFIC_MC_KEYWORDS,
    TRAFFIC_TEXT_DIR,
    TRAFFIC_TITLE_KEYWORDS,
)


MIN_LOCAL_DOC_CHARS = 500
ARTICLE_RE = re.compile(r"(Điều\s+\d+[a-zA-Z]?\.\s*[^\n]{0,180})")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks for legal text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def is_traffic_title(title: str) -> bool:
    return contains_keyword(title, TRAFFIC_TITLE_KEYWORDS)


def is_traffic_text(text: str) -> bool:
    return contains_keyword(text, TRAFFIC_CONTENT_KEYWORDS)


def is_traffic_mc_item(item: dict) -> bool:
    choices = item.get("choices") or []
    joined = " ".join([item.get("question", ""), *choices])
    return contains_keyword(joined, TRAFFIC_MC_KEYWORDS)


def infer_article(text: str) -> str:
    match = ARTICLE_RE.search(text)
    return compact_text(match.group(1)) if match else ""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def load_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {path}. "
            "Create docs/docs_giaothong/manifest.json before building the corpus."
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    policy = manifest.get("source_policy")
    if policy != KB_SOURCE_POLICY:
        raise ValueError(
            f"Manifest source_policy must be {KB_SOURCE_POLICY!r}, got {policy!r}."
        )

    docs = manifest.get("documents")
    if not isinstance(docs, list) or not docs:
        raise ValueError("Manifest must contain a non-empty 'documents' list.")

    return manifest


def iter_manifest_documents(enabled_only: bool = True) -> Iterator[dict]:
    manifest = load_source_manifest()
    text_root = TRAFFIC_TEXT_DIR.resolve()

    seen_doc_ids: set[str] = set()
    for raw in manifest["documents"]:
        if enabled_only and not raw.get("enabled", True):
            continue

        doc_id = str(raw.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(f"Manifest entry is missing doc_id: {raw}")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id in manifest: {doc_id}")
        seen_doc_ids.add(doc_id)

        source_path = str(raw.get("source_path") or "").strip()
        if not source_path:
            raise ValueError(f"Manifest entry {doc_id} is missing source_path.")

        abs_path = (PROJECT_ROOT / source_path).resolve()
        if not _is_relative_to(abs_path, text_root):
            raise ValueError(
                f"Manifest entry {doc_id} points outside {TRAFFIC_TEXT_DIR}: {source_path}"
            )
        if abs_path.suffix.lower() != ".txt":
            raise ValueError(f"Manifest entry {doc_id} must point to a .txt file: {source_path}")
        if not abs_path.exists():
            raise FileNotFoundError(f"Manifest entry {doc_id} file not found: {abs_path}")

        item = dict(raw)
        item["path"] = abs_path
        item["source_path"] = source_path
        yield item


def iter_local_traffic_records(shuffle_seed: int | None = None) -> Iterator[dict]:
    docs = list(iter_manifest_documents(enabled_only=True))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(docs)

    for item in docs:
        path = item["path"]
        text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < MIN_LOCAL_DOC_CHARS:
            raise ValueError(
                f"Local text is too short to be a trusted legal source "
                f"({len(text)} chars): {path}"
            )

        title = str(item.get("title") or path.stem)
        if not is_traffic_title(title) and not is_traffic_text(text):
            raise ValueError(
                f"Enabled manifest document does not look traffic-related: {item['doc_id']}"
            )

        yield {
            "text": text,
            "source": item["doc_id"],
            "doc_id": item["doc_id"],
            "title": title,
            "authority": item.get("authority") or "",
            "document_type": item.get("document_type") or "",
            "effective_date": item.get("effective_date"),
            "source_path": item["source_path"],
            "corpus": "local_text",
            "source_policy": KB_SOURCE_POLICY,
        }


def iter_traffic_corpus_records(shuffle_seed: int | None = None) -> Iterator[dict]:
    yield from iter_local_traffic_records(shuffle_seed=shuffle_seed)


def iter_traffic_documents(shuffle_seed: int | None = None) -> Iterator[Document]:
    for record in iter_traffic_corpus_records(shuffle_seed=shuffle_seed):
        yield Document(
            page_content=record["text"],
            metadata={
                "source": record["source"],
                "doc_id": record["doc_id"],
                "title": record["title"],
                "authority": record["authority"],
                "document_type": record["document_type"],
                "effective_date": record["effective_date"],
                "source_path": record["source_path"],
                "corpus": record["corpus"],
                "source_policy": record["source_policy"],
            },
        )
