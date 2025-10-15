from __future__ import annotations

import hashlib
import re
import warnings
from typing import Iterable, List

import numpy as np

from config import Config


class Embedder:
    """Embedding provider using FlagEmbedding/BGEM3FlagModel with hashing fallback."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._use_hf = False
        self.model = None

        if cfg.embed_use_hf:
            try:
                from FlagEmbedding import BGEM3FlagModel
                # The model will download automatically on first use.
                self.model = BGEM3FlagModel(cfg.embed_model_name, use_fp16=cfg.embed_use_fp16)
                self._use_hf = True
                print("BGEM3FlagModel loaded successfully.")
            except ImportError:
                warnings.warn("'FlagEmbedding' package not found. Falling back to heuristic hashing.")
                self._use_hf = False
            except Exception as e:
                warnings.warn(f"Failed to load BGEM3FlagModel: {e}. Falling back to heuristic hashing.")
                self._use_hf = False

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def embed_text(self, text: str) -> List[float]:
        if self._use_hf and self.model:
            dense_vecs = self.model.encode(
                [text],
                batch_size=1,
                max_length=512  # A reasonable default for performance
            )['dense_vecs']
            normalized_vec = self._normalize(dense_vecs[0])
            return normalized_vec.astype(np.float32).tolist()
        
        # Fallback to hashing
        return self._hash_embed(text)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        if self._use_hf and self.model:
            text_list = list(texts)
            if not text_list:
                return []
            dense_vecs = self.model.encode(
                text_list,
                batch_size=12,
                max_length=512
            )['dense_vecs']
            
            # Normalize each vector in the resulting matrix
            norm = np.linalg.norm(dense_vecs, axis=1, keepdims=True) + 1e-12
            normalized_vecs = dense_vecs / norm
            
            return normalized_vecs.astype(np.float32).tolist()

        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> List[float]:
        d = self.cfg.embed_dimension
        vec = np.zeros(d, dtype=np.float64)
        tokens = self._tokenize(text)
        # unigrams and bigrams
        for i, tok in enumerate(tokens):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % d
            vec[idx] += 1.0
            if i + 1 < len(tokens):
                bg = tok + "_" + tokens[i + 1]
                h2 = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
                idx2 = h2 % d
                vec[idx2] += 0.5
        vec = self._normalize(vec)
        return vec.astype(np.float32).tolist()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fa5\s]+", " ", text)
        return [t for t in text.split() if t]
