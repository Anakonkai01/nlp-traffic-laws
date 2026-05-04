import argparse
import gc
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.5"

import faiss
import numpy as np
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FORBIDDEN_LEGACY_TERMS,
    KB_META_PATH,
    KB_PATH,
    KB_SOURCE_POLICY,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    SEPARATORS,
)
from corpus import infer_article, iter_manifest_documents, iter_traffic_documents


EMBED_MODEL = "BAAI/bge-m3"
EMBED_BATCH = 128
EMBED_MAX_SEQ = 1024

FORBIDDEN_LEGACY_TERMS_LOWER = [term.lower() for term in FORBIDDEN_LEGACY_TERMS]
LEGAL_CHUNKING_POLICY = "legal_article_clause_v1"

ARTICLE_RE = re.compile(r"(?m)^Điều\s+(\d+[a-zA-Z]?)\.\s*[^\n]+")
CLAUSE_RE = re.compile(r"(?m)^(\d+)\.\s+")


def _read_build_meta() -> dict:
    if not KB_META_PATH.exists():
        raise FileNotFoundError(
            f"Missing KB metadata: {KB_META_PATH}. "
            "This vector store is stale or was not built by the local-text-only pipeline."
        )
    meta = json.loads(KB_META_PATH.read_text(encoding="utf-8"))
    if meta.get("source_policy") != KB_SOURCE_POLICY:
        raise ValueError(
            f"KB source_policy must be {KB_SOURCE_POLICY!r}, got {meta.get('source_policy')!r}."
        )
    return meta


def assert_local_text_only_kb() -> dict:
    if not KB_PATH.exists() or not any(KB_PATH.iterdir()):
        raise FileNotFoundError(f"Traffic vector DB not found at {KB_PATH}. Run python src/build_kb.py.")
    return _read_build_meta()


def _manifest_snapshot() -> list[dict]:
    snapshot = []
    for item in iter_manifest_documents(enabled_only=True):
        path = item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot.append(
            {
                "doc_id": item["doc_id"],
                "title": item.get("title", ""),
                "source_path": item["source_path"],
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return snapshot


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _contains_forbidden_legacy(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in FORBIDDEN_LEGACY_TERMS_LOWER)


def _use_cached_hf_models_only() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass
    try:
        import transformers.utils.hub as hf_hub

        if hasattr(hf_hub, "_is_offline_mode"):
            hf_hub._is_offline_mode = True
    except Exception:
        pass


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fallback_splitter() -> RecursiveCharacterTextSplitter:
    separators = [*SEPARATORS]
    if "" not in separators:
        separators.append("")
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )


def _split_long_legal_chunk(chunk: Document) -> list[Document]:
    """Split unusually long legal blocks while preserving article context."""
    text = chunk.page_content
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk]

    metadata = dict(chunk.metadata)
    article = metadata.get("article") or infer_article(text)
    parts = []
    for idx, part in enumerate(_fallback_splitter().split_text(text), start=1):
        content = part.strip()
        if article and article not in content[: len(article) + 20]:
            content = f"{article}\n\n{content}"
        part_md = dict(metadata)
        part_md["legal_split_part"] = idx
        parts.append(Document(page_content=content, metadata=part_md))
    return parts


def _article_documents(doc: Document) -> list[Document]:
    text = doc.page_content
    article_matches = list(ARTICLE_RE.finditer(text))
    if not article_matches:
        return _fallback_splitter().split_documents([doc])

    chunks: list[Document] = []
    base_metadata = dict(doc.metadata)

    preamble = text[: article_matches[0].start()].strip()
    if preamble:
        chunks.append(Document(page_content=preamble, metadata={**base_metadata, "legal_section": "preamble"}))

    for article_idx, match in enumerate(article_matches):
        article_no = match.group(1)
        article_title = _compact(match.group(0))
        article_start = match.start()
        article_end = article_matches[article_idx + 1].start() if article_idx + 1 < len(article_matches) else len(text)
        article_text = text[article_start:article_end].strip()
        clause_matches = list(CLAUSE_RE.finditer(article_text))
        article_metadata = {
            **base_metadata,
            "article": article_title,
            "article_number": article_no,
            "legal_section": "article",
            "chunking_policy": LEGAL_CHUNKING_POLICY,
        }

        if not clause_matches:
            chunks.extend(_split_long_legal_chunk(Document(page_content=article_text, metadata=article_metadata)))
            continue

        lead = article_text[: clause_matches[0].start()].strip()
        if lead and _compact(lead) != article_title:
            lead_doc = Document(page_content=lead, metadata={**article_metadata, "legal_block": "article_lead"})
            chunks.extend(_split_long_legal_chunk(lead_doc))

        for clause_idx, clause_match in enumerate(clause_matches):
            clause_no = clause_match.group(1)
            clause_start = clause_match.start()
            clause_end = clause_matches[clause_idx + 1].start() if clause_idx + 1 < len(clause_matches) else len(article_text)
            clause_text = article_text[clause_start:clause_end].strip()
            content = f"{article_title}\n\n{clause_text}"
            clause_doc = Document(
                page_content=content,
                metadata={
                    **article_metadata,
                    "clause_number": clause_no,
                    "legal_block": "clause",
                },
            )
            chunks.extend(_split_long_legal_chunk(clause_doc))

    return chunks


def _enrich_chunk(chunk: Document, ordinal: int) -> Document:
    metadata = dict(chunk.metadata)
    metadata["article"] = metadata.get("article") or infer_article(chunk.page_content)
    metadata["chunk_id"] = f"{metadata.get('doc_id', 'unknown')}:{ordinal:06d}"
    metadata["source_policy"] = KB_SOURCE_POLICY
    return Document(page_content=chunk.page_content, metadata=metadata)


def build_kb(force: bool = False) -> None:
    if KB_PATH.exists() and any(KB_PATH.iterdir()):
        if force:
            shutil.rmtree(KB_PATH)
        else:
            meta = assert_local_text_only_kb()
            print(
                f"KB already exists at {KB_PATH} "
                f"({meta.get('chunk_count', '?')} chunks, policy={meta.get('source_policy')})."
            )
            print("Use --force to rebuild it from the current manifest.")
            return

    docs = list(iter_traffic_documents())
    if not docs:
        raise RuntimeError("No enabled local text documents found in the source manifest.")

    _use_cached_hf_models_only()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBED_MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
    )
    model.eval()
    model.max_seq_length = EMBED_MAX_SEQ
    dim = model.get_embedding_dimension()

    index = faiss.IndexFlatIP(dim)
    all_docs: list[Document] = []
    buffer: list[Document] = []
    seen_hashes: set[str] = set()
    stats = {"total": 0, "dup": 0, "short": 0, "long": 0, "legacy": 0, "added": 0}
    chunk_ordinal = 0

    def _keep(chunk: Document) -> bool:
        n = len(chunk.page_content)
        if n < MIN_CHUNK_CHARS:
            stats["short"] += 1
            return False
        if n > MAX_CHUNK_CHARS:
            stats["long"] += 1
            return False
        if _contains_forbidden_legacy(chunk.page_content):
            stats["legacy"] += 1
            return False
        h = _chunk_hash(chunk.page_content)
        if h in seen_hashes:
            stats["dup"] += 1
            return False
        seen_hashes.add(h)
        return True

    def flush(buf: list[Document]) -> None:
        texts = [c.page_content for c in buf]
        with torch.inference_mode():
            vecs = model.encode(
                texts,
                batch_size=EMBED_BATCH,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        index.add(np.array(vecs, dtype=np.float32))
        all_docs.extend(buf)
        stats["added"] += len(buf)
        del vecs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar = tqdm(docs, desc="Building local traffic KB", unit="doc")
    for doc in pbar:
        for raw_chunk in _article_documents(doc):
            stats["total"] += 1
            if not _keep(raw_chunk):
                continue
            chunk_ordinal += 1
            buffer.append(_enrich_chunk(raw_chunk, chunk_ordinal))
        if len(buffer) >= EMBED_BATCH:
            flush(buffer)
            buffer.clear()

    if buffer:
        flush(buffer)

    if index.ntotal == 0:
        raise RuntimeError("No chunks passed quality gates; KB was not created.")

    print(
        f"Chunks: {stats['total']} total | "
        f"{stats['dup']} dup | {stats['short']} short | "
        f"{stats['long']} oversized | {stats['added']} embedded"
        f" | {stats['legacy']} legacy-filtered"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _use_cached_hf_models_only()
    from langchain_huggingface import HuggingFaceEmbeddings

    lc_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH},
    )
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_docs)})
    index_to_id = {i: str(i) for i in range(len(all_docs))}
    vectorstore = FAISS(
        embedding_function=lc_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )

    KB_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(KB_PATH))
    meta = {
        "source_policy": KB_SOURCE_POLICY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBED_MODEL,
        "embedding_max_seq": EMBED_MAX_SEQ,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_policy": LEGAL_CHUNKING_POLICY,
        "chunk_count": index.ntotal,
        "source_documents": _manifest_snapshot(),
    }
    KB_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {index.ntotal:,} chunks -> {KB_PATH}")


def load_vectorstore() -> FAISS:
    assert_local_text_only_kb()
    _use_cached_hf_models_only()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH},
    )
    _st = getattr(embeddings, "_client", None) or getattr(embeddings, "client", None)
    if _st is not None:
        _st.max_seq_length = EMBED_MAX_SEQ
    return FAISS.load_local(
        str(KB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the existing local-text KB.")
    args = parser.parse_args()
    build_kb(force=args.force)


if __name__ == "__main__":
    main()
