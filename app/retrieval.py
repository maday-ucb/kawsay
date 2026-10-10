"""
Semantic search with multilingual embeddings vs BM25 baseline.

SearchEngine.search(query) returns both rankings so the frontend
and evaluation can compare them side by side.
"""
import json
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.documents import corpus_hash, read_chunks
from app.settings import ROOT, MODEL_DIR, ARTIFACT_DIR, TOP_K, EMBEDDING_ID


# ── Encoder ──────────────────────────────────────────────────────────────────

def load_encoder(local_only: bool = False) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if local_only and (MODEL_DIR / "embedding").exists():
        return SentenceTransformer(
            str(MODEL_DIR / "embedding"), device=device, local_files_only=True
        )
    encoder = SentenceTransformer(EMBEDDING_ID, device=device)
    encoder.save(str(MODEL_DIR / "embedding"))
    return encoder


def encode_query(encoder: SentenceTransformer, query: str) -> np.ndarray:
    """multilingual-e5-base requires 'query: ' prefix for queries."""
    return encoder.encode(
        [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
    )[0]


def encode_passages(encoder: SentenceTransformer, texts: list[str],
                    batch_size: int = 32) -> np.ndarray:
    """multilingual-e5-base requires 'passage: ' prefix for documents."""
    prefixed = [f"passage: {t}" for t in texts]
    return encoder.encode(
        prefixed, normalize_embeddings=True,
        show_progress_bar=True, batch_size=batch_size,
        convert_to_numpy=True,
    )


# ── Ranking helpers ───────────────────────────────────────────────────────────

def ranked_indices(scores: np.ndarray, chunk_ids: list[str],
                   k: int = TOP_K) -> list[int]:
    """Return top-k indices sorted by score descending; ties broken by chunk_id."""
    return sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), chunk_ids[i])
    )[:k]


# ── Search engine ─────────────────────────────────────────────────────────────

class SearchEngine:
    def __init__(self, encoder: SentenceTransformer | None = None):
        index_path = ARTIFACT_DIR / "index.joblib"
        if not index_path.exists():
            raise RuntimeError(
                "Index not found. Run: python -m app.build_index"
            )

        self.index = joblib.load(index_path)

        # Verify corpus has not changed since indexing
        current_hash = corpus_hash(read_chunks())
        if self.index["corpus_hash"] != current_hash:
            raise RuntimeError(
                "Corpus changed since last index build. "
                "Run: python -m app.build_index"
            )

        self.encoder   = encoder or load_encoder(local_only=True)
        self.chunks    = self.index["chunks"]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]

    def search(self, query: str, k: int = TOP_K) -> dict:
        # ── Semantic search ──
        q_vector        = encode_query(self.encoder, query)
        semantic_scores = self.index["embeddings"] @ q_vector

        # ── BM25 baseline ──
        query_tokens  = query.lower().split()
        bm25_scores   = np.array(self.index["bm25"].get_scores(query_tokens))

        results = {}
        for method, scores in [("semantic", semantic_scores), ("lexical", bm25_scores)]:
            top = ranked_indices(scores, self.chunk_ids, k)
            results[method] = [
                {**self.chunks[i], "score": float(scores[i])}
                for i in top
            ]

        return results
