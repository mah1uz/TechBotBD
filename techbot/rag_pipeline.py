"""
RAG pipeline: FAISS vector store + retrieval.

Core NLP loop in ~60 lines:
  1. Build a search_text per product (concatenation of relevant spec columns)
  2. Embed all products with sentence-transformers
  3. Index with FAISS IndexFlatIP (cosine similarity since vectors are normalized)
  4. For a query: embed it, search index, dedup by product name (cheapest variant)

Notes:
  - We use cosine similarity (not L2 distance) so the score is in [-1, 1].
    Higher = better. Range in practice: 0.3 (weak) to 0.9 (strong).
  - The E5 family of models requires "query: " / "passage: " prefixes
    on inputs; we detect this from the model name and prefix automatically.
  - No disk cache — Streamlit's @st.cache_resource handles in-session caching.
    First build is ~5 seconds for 413 products; subsequent reruns are free.
"""

from __future__ import annotations

from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from techbot.config import (
    EMBEDDING_MODEL, RETRIEVAL_TOP_K,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
)
from techbot.preprocessor import preprocess


# ---------------------------------------------------------------------------
# search_text builder
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    """Return non-empty string, or '' for nan/None/'N/A'/'nan'."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "n/a", "none") else s


def _row_to_search_text(row: pd.Series) -> str:
    """Concatenate spec columns into one searchable sentence."""
    parts = []
    for col in ("name", "category", "sub_category", "brand",
                "processor", "generation", "gpu"):
        v = _clean(row.get(col))
        if v and v.lower() != "unknown":
            parts.append(v)
    # Numeric specs — only if non-zero
    if (ram := row.get("ram_gb", 0)) and float(ram) > 0:
        parts.append(f"{int(float(ram))}GB RAM")
    if (storage := _clean(row.get("storage"))):
        parts.append(f"{storage} storage")
    if (disp := row.get("display_inch", 0)) and float(disp) > 0:
        parts.append(f"{disp} inch")
    if (bat := row.get("battery_mah", 0)) and float(bat) > 0:
        parts.append(f"{int(float(bat))}mAh battery")
    if (cam := row.get("camera_mp", 0)) and float(cam) > 0:
        parts.append(f"{int(float(cam))}MP camera")
    return " ".join(parts)


def _is_e5_model(model_name: str) -> bool:
    """E5 models need 'query:' / 'passage:' prefixes on inputs."""
    return "e5" in model_name.lower()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TechBotRAG:
    """
    FAISS-backed retriever over a product DataFrame.

    Single responsibility: given a query string, return the top-K most
    semantically similar products as dicts.
    """

    def __init__(self, df: pd.DataFrame, **_unused):
        if df is None or len(df) == 0:
            raise ValueError("TechBotRAG requires a non-empty DataFrame.")
        self.df = df.reset_index(drop=True).copy()
        self.df["search_text"] = self.df.apply(_row_to_search_text, axis=1)
        self.use_e5_prefix = _is_e5_model(EMBEDDING_MODEL)

        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

        print(f"[RAG] Encoding {len(self.df)} products...")
        embeddings = self._encode(self.df["search_text"].tolist(), is_query=False)
        # IndexFlatIP on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"[RAG] Index ready ({self.dim}-dim).")

    # ----- encoding -----
    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Embed texts, normalized for cosine similarity. Add E5 prefix if needed."""
        if self.use_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")

    # ----- retrieval -----
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Embed query, search FAISS, dedup by product name (keeping cheapest
        variant). Returns top_k results with `faiss_score` field added.
        """
        k = top_k or RETRIEVAL_TOP_K
        # Over-fetch since dedup may collapse multiple rows of the same model
        fetch_k = min(k * 4, len(self.df))

        clean_query = preprocess(query) or query
        q_emb = self._encode([clean_query], is_query=True)
        scores, idxs = self.index.search(q_emb, fetch_k)

        seen: dict[str, dict] = {}
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx].to_dict()
            row["faiss_score"] = float(score)
            row.pop("search_text", None)

            name_key = _clean(row.get("name")).lower()
            if not name_key:
                continue

            existing = seen.get(name_key)
            if existing is None:
                seen[name_key] = row
            else:
                # Keep cheaper variant (price 0 = unknown -> treat as worst)
                cur = row.get("price_bdt", 0) or 10**12
                old = existing.get("price_bdt", 0) or 10**12
                if cur < old:
                    seen[name_key] = row

            if len(seen) >= k:
                break

        return list(seen.values())[:k]

    @staticmethod
    def get_confidence(faiss_score: float) -> str:
        """Map cosine similarity to a discrete confidence label."""
        if faiss_score >= CONFIDENCE_HIGH:
            return "high"
        if faiss_score >= CONFIDENCE_MEDIUM:
            return "medium"
        return "low"
