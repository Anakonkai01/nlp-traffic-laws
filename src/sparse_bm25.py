"""
BM25 sparse retriever over pyvi-tokenized FAISS chunks.

CLI:
    python src/sparse_bm25.py build  # rebuild bm25.pkl from current vector_db_traffic
    python src/sparse_bm25.py stats  # show summary

Library:
    bm = BM25Retriever.load_or_build(vs)
    bm.search(question, k=50)  # → list[(doc, score)]

Index file: vector_db_traffic/bm25.pkl  (rebuilt when KB chunk count changes).
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

from pyvi import ViTokenizer  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore

from config import KB_PATH

BM25_PATH = KB_PATH / "bm25.pkl"

# Lower-cased Vietnamese token splitter (after pyvi merges multi-syllable terms).
_TOKEN_RE = re.compile(r"[\wÀ-ỹ_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase + pyvi underscore-segmented tokenization."""
    if not text:
        return []
    seg = ViTokenizer.tokenize(text.lower())
    return _TOKEN_RE.findall(seg)


class BM25Retriever:
    def __init__(self, docs, tokenized: list[list[str]], bm25: BM25Okapi):
        self._docs = docs
        self._tokenized = tokenized
        self._bm25 = bm25

    @classmethod
    def build(cls, vs) -> "BM25Retriever":
        docstore = getattr(vs, "docstore", None)
        docs = list(getattr(docstore, "_dict", {}).values()) if docstore else []
        if not docs:
            raise RuntimeError("vectorstore has no docs in docstore")
        tokenized = [tokenize(d.page_content or "") for d in docs]
        bm25 = BM25Okapi(tokenized)
        return cls(docs, tokenized, bm25)

    def save(self, path: Path = BM25_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "docs": self._docs,
                    "tokenized": self._tokenized,
                    "n_docs": len(self._docs),
                },
                f,
            )

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Retriever":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        bm25 = BM25Okapi(blob["tokenized"])
        return cls(blob["docs"], blob["tokenized"], bm25)

    @classmethod
    def load_or_build(cls, vs, path: Path = BM25_PATH) -> "BM25Retriever":
        if path.exists():
            try:
                inst = cls.load(path)
                if len(inst._docs) == vs.index.ntotal:
                    return inst
            except Exception:
                pass
        inst = cls.build(vs)
        inst.save(path)
        return inst

    def search(self, query: str, k: int = 50):
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if k >= len(scores):
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
        else:
            import numpy as np

            order = np.argpartition(-scores, k)[:k]
            order = sorted(order, key=lambda i: -scores[i])
        return [(self._docs[i], float(scores[i])) for i in order]

    def __len__(self):
        return len(self._docs)


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats", "search"])
    ap.add_argument("--query", default=None)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "build":
        from build_kb import load_vectorstore

        vs = load_vectorstore()
        bm = BM25Retriever.build(vs)
        bm.save()
        print(f"[ok] BM25 built · {len(bm)} docs · path={BM25_PATH}")
        return

    if args.cmd == "stats":
        bm = BM25Retriever.load()
        print(f"[stats] n_docs={len(bm)} · path={BM25_PATH}")
        return

    if args.cmd == "search":
        bm = BM25Retriever.load()
        if not args.query:
            print("--query required for 'search'", file=sys.stderr)
            sys.exit(2)
        for doc, sc in bm.search(args.query, k=args.k):
            md = doc.metadata or {}
            print(f"[{sc:7.3f}] {md.get('doc_id')} · Đ.{md.get('article_number') or '?'}")
        return


if __name__ == "__main__":
    _cli()
