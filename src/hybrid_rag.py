"""Local Hybrid RAG: deterministic dense embeddings + BM25 + RRF fusion."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

LOCAL_EMBEDDING_DIM = 384
DEFAULT_DB_PATH = "db/deal_desk.db"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and dense embedder."""
    return _TOKEN_RE.findall(text.lower())


def local_embed_text(text: str, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Deterministic local dense embedding (hashing bag-of-words, L2-normalized)."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = tokenize(text)
    if not tokens:
        return vec.tolist()
    for token in tokens:
        # Stable FNV-1a style hash → bucket + signed contribution
        h = 2166136261
        for ch in token:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_chunks(
    conn: sqlite3.Connection,
    source_type: str | None = None,
) -> list[sqlite3.Row]:
    if source_type:
        return list(
            conn.execute(
                "SELECT * FROM rag_chunks WHERE source_type = ? ORDER BY chunk_id",
                (source_type,),
            )
        )
    return list(conn.execute("SELECT * FROM rag_chunks ORDER BY chunk_id"))


def ensure_embeddings(db_path: str = DEFAULT_DB_PATH) -> int:
    """Populate embedding_json for any rag_chunks row where it is NULL. Returns count updated."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT chunk_id, content FROM rag_chunks WHERE embedding_json IS NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        emb = json.dumps(local_embed_text(row["content"]))
        conn.execute(
            "UPDATE rag_chunks SET embedding_json = ? WHERE chunk_id = ?",
            (emb, row["chunk_id"]),
        )
        updated += 1
    conn.commit()
    conn.close()
    return updated


def _row_to_result(row: sqlite3.Row, score: float) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "source_type": row["source_type"],
        "category": row["category"],
        "title": row["title"],
        "content": row["content"],
        "score": float(score),
    }


def bm25_only_search(
    query: str,
    top_k: int = 4,
    source_type: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Lexical BM25 retrieval (v1 baseline)."""
    conn = _connect(db_path)
    chunks = _load_chunks(conn, source_type)
    conn.close()
    if not chunks:
        return []

    corpus_tokens = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score <= 0 and len(results) > 0:
            # Keep trailing zero-score only if nothing better; still return top_k slots
            pass
        results.append(_row_to_result(chunks[int(idx)], score))
    return results


def hybrid_rrf_search(
    query: str,
    top_k: int = 4,
    source_type: str | None = None,
    rrf_k: int = 60,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """BM25 + dense cosine fused via Reciprocal Rank Fusion (v2/v3)."""
    ensure_embeddings(db_path)
    conn = _connect(db_path)
    chunks = _load_chunks(conn, source_type)
    conn.close()
    if not chunks:
        return []

    query_tokens = tokenize(query)
    corpus_tokens = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_rank = np.argsort(bm25_scores)[::-1]

    query_vec = np.array(local_embed_text(query), dtype=np.float64)
    dense_scores = []
    for chunk in chunks:
        emb_raw = chunk["embedding_json"]
        if not emb_raw:
            dense_scores.append(0.0)
            continue
        doc_vec = np.array(json.loads(emb_raw), dtype=np.float64)
        dense_scores.append(cosine_similarity(query_vec, doc_vec))
    dense_scores_arr = np.array(dense_scores, dtype=np.float64)
    dense_rank = np.argsort(dense_scores_arr)[::-1]

    rrf_scores = np.zeros(len(chunks), dtype=np.float64)
    for rank, idx in enumerate(bm25_rank, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(dense_rank, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)

    top_idx = np.argsort(rrf_scores)[::-1][:top_k]
    return [_row_to_result(chunks[int(i)], float(rrf_scores[int(i)])) for i in top_idx]
