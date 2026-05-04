import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_CHARS, SEPARATORS
from corpus import infer_article

ARTICLE_RE = re.compile(r"(?m)^Điều\s+(\d+[a-zA-Z]?)\.\s*[^\n]+")
CLAUSE_RE = re.compile(r"(?m)^(\d+)\.\s+")


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


def _split_long_chunk(chunk: Document) -> list[Document]:
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


def article_clause_chunks(doc: Document) -> list[Document]:
    """Split a Document into article/clause-level chunks for QA generation and KB indexing."""
    text = doc.page_content
    article_matches = list(ARTICLE_RE.finditer(text))
    if not article_matches:
        return _fallback_splitter().split_documents([doc])

    chunks: list[Document] = []
    base_metadata = dict(doc.metadata)

    preamble = text[: article_matches[0].start()].strip()
    if preamble:
        chunks.append(
            Document(page_content=preamble, metadata={**base_metadata, "legal_section": "preamble"})
        )

    for article_idx, match in enumerate(article_matches):
        article_no = match.group(1)
        article_title = _compact(match.group(0))
        article_start = match.start()
        article_end = (
            article_matches[article_idx + 1].start()
            if article_idx + 1 < len(article_matches)
            else len(text)
        )
        article_text = text[article_start:article_end].strip()
        clause_matches = list(CLAUSE_RE.finditer(article_text))
        article_metadata = {
            **base_metadata,
            "article": article_title,
            "article_number": article_no,
            "legal_section": "article",
        }

        if not clause_matches:
            chunks.extend(
                _split_long_chunk(Document(page_content=article_text, metadata=article_metadata))
            )
            continue

        lead = article_text[: clause_matches[0].start()].strip()
        if lead and _compact(lead) != article_title:
            chunks.extend(
                _split_long_chunk(
                    Document(
                        page_content=lead,
                        metadata={**article_metadata, "legal_block": "article_lead"},
                    )
                )
            )

        for clause_idx, clause_match in enumerate(clause_matches):
            clause_no = clause_match.group(1)
            clause_start = clause_match.start()
            clause_end = (
                clause_matches[clause_idx + 1].start()
                if clause_idx + 1 < len(clause_matches)
                else len(article_text)
            )
            clause_text = article_text[clause_start:clause_end].strip()
            content = f"{article_title}\n\n{clause_text}"
            clause_doc = Document(
                page_content=content,
                metadata={**article_metadata, "clause_number": clause_no, "legal_block": "clause"},
            )
            chunks.extend(_split_long_chunk(clause_doc))

    return chunks
