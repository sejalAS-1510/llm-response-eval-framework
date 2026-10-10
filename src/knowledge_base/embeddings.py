"""
Turns chunk text into embeddings (dimension: 384).
Engineered with an ultra-low-memory mode (< 1 MB RAM) for cloud deployments
with strict memory constraints (such as Render's 512 MB Free tier).
"""

import hashlib
import logging
import math
import os
import re
from typing import Any, List

logger = logging.getLogger(__name__)

DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _lightweight_semantic_embed(text: str) -> List[float]:
    """
    Ultra-low-memory deterministic embedding generator (384 dimensions).
    Uses subword hashing and character n-gram projections normalized to unit length.
    Consumes < 0.1 MB of RAM, preventing Out-Of-Memory (OOM) kills on cloud free tiers.
    """
    if not text:
        return [0.0] * DIMENSION

    vec = [0.0] * DIMENSION
    clean_text = text.lower()
    tokens = re.findall(r"\w+", clean_text)

    # Word-level feature hashing
    for token in tokens:
        h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        h2 = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx1 = h1 % DIMENSION
        idx2 = h2 % DIMENSION
        sign1 = 1.0 if ((h1 >> 16) & 1) else -1.0
        sign2 = 1.0 if ((h2 >> 16) & 1) else -1.0
        vec[idx1] += sign1
        vec[idx2] += sign2 * 0.5

    # Subword/character 3-gram features (captures morphology and keyword overlap)
    for i in range(len(clean_text) - 2):
        trigram = clean_text[i : i + 3]
        h = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
        idx = h % DIMENSION
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign * 0.3

    # Normalize to unit length (L2 norm)
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-9:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0

    return vec


def get_model() -> Any:
    """
    Optionally loads local heavy SentenceTransformer only if explicitly enabled
    via USE_LOCAL_TRANSFORMERS=true. In standard cloud/container mode, returns None
    to preserve memory well within 512 MB.
    """
    global _model
    if os.getenv("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
            except Exception as e:
                logger.warning(
                    f"Could not load local SentenceTransformer: {e}. Using lightweight engine."
                )
                _model = None
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    if model is not None:
        try:
            return model.encode(texts, show_progress_bar=False).tolist()
        except Exception as e:
            logger.warning(
                f"SentenceTransformer encoding failed: {e}. Falling back to lightweight embeddings."
            )

    return [_lightweight_semantic_embed(t) for t in texts]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
