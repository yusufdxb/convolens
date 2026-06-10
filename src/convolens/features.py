"""Text feature extraction: TF-IDF and cached sentence embeddings."""
from __future__ import annotations

import hashlib
import os

import numpy as np

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", ".cache")


def _cache_path(texts: list[str]) -> str:
    key = hashlib.sha1(("\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"emb_{_EMBED_MODEL.split('/')[-1]}_{len(texts)}_{key}.npy")


def embed(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """Sentence-embed texts with MiniLM, caching the result to disk.

    Embeddings are deterministic for a given input set, so we content-address the
    cache and skip recompute on reruns.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(texts)
    if use_cache and os.path.exists(path):
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_EMBED_MODEL)
    vecs = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    if use_cache:
        np.save(path, vecs)
    return vecs
