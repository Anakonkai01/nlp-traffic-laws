import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_CHARS, SEPARATORS, STRUCT_ARTICLE_RE, STRUCT_CLAUSE_RE, STRUCT_POINT_RE
from corpus import infer_article

# Emit point chunks only for clauses that list at least this many points.
# Short clauses (a/b only) are well-served by the clause chunk already.
POINT_CHUNK_MIN_POINTS = 2
# Don't split clauses shorter than this — they already fit a single answer.
POINT_CHUNK_MIN_CLAUSE_CHARS = 400


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
    article_matches = list(STRUCT_ARTICLE_RE.finditer(text))
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
        clause_matches = list(STRUCT_CLAUSE_RE.finditer(article_text))
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

            # Emit additional point-level chunks for clauses dense in points.
            # Each point chunk preserves the article title and the clause lead
            # (which contains the sanction headline, e.g. "Phạt tiền từ ...")
            # so retrieval can match question terms against a single violation.
            point_matches = list(STRUCT_POINT_RE.finditer(clause_text))
            if (
                len(point_matches) >= POINT_CHUNK_MIN_POINTS
                and len(clause_text) >= POINT_CHUNK_MIN_CLAUSE_CHARS
            ):
                clause_lead = clause_text[: point_matches[0].start()].strip()
                for point_idx, point_match in enumerate(point_matches):
                    point_letter = point_match.group(1)
                    point_start = point_match.start()
                    point_end = (
                        point_matches[point_idx + 1].start()
                        if point_idx + 1 < len(point_matches)
                        else len(clause_text)
                    )
                    point_text = clause_text[point_start:point_end].strip()
                    point_content_parts = [article_title]
                    if clause_lead:
                        point_content_parts.append(f"Khoản {clause_no}. {clause_lead}")
                    else:
                        point_content_parts.append(f"Khoản {clause_no}.")
                    point_content_parts.append(point_text)
                    point_doc = Document(
                        page_content="\n\n".join(point_content_parts),
                        metadata={
                            **article_metadata,
                            "clause_number": clause_no,
                            "point_letter": point_letter,
                            "legal_block": "point",
                        },
                    )
                    chunks.extend(_split_long_chunk(point_doc))

    return chunks
