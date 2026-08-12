#!/usr/bin/env python3
"""
reranker.py - Cross-encoder reranker for the HUST RAG retriever.

Default model:
    BAAI/bge-reranker-v2-m3

The reranker receives the query and a candidate list produced by hybrid search,
then reorders candidates by cross-encoder relevance score.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return the best available device for local reranking."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class RerankerConfig:
    """Configuration for a local cross-encoder reranker."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str | None = None
    max_length: int = 512
    batch_size: int = 16


class BGEReranker:
    """
    Lazy-loading reranker backed by sentence-transformers CrossEncoder.

    The class intentionally avoids importing RetrievalResult to keep this module
    independent from retriever.py and easy to reuse in tests or other pipelines.
    """

    def __init__(self, config: RerankerConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")
        self.config = config or RerankerConfig(**kwargs)
        self._model = None

    @property
    def model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        device = self.config.device or detect_device()
        logger.info("Loading reranker model: %s -> %s", self.config.model_name, device)
        self._model = CrossEncoder(
            self.config.model_name,
            device=device,
            max_length=self.config.max_length,
        )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return relevance scores for query/passage pairs."""
        if not passages:
            return []

        pairs = [(query, passage or "") for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        results: Sequence[Any],
        *,
        top_k: int | None = None,
    ) -> list[Any]:
        """
        Reorder retrieval results by reranker score.

        Each returned item is a shallow copy of the original result with:
          - score set to rerank_score
          - metadata["rerank_score"] added
          - metadata["pre_rerank_score"] preserving the previous score
        """
        if not results:
            return []

        scores = self.score(query, [getattr(result, "text", "") for result in results])
        reranked: list[Any] = []

        for result, rerank_score in zip(results, scores):
            item = copy.copy(result)
            metadata = dict(getattr(result, "metadata", {}) or {})
            metadata["pre_rerank_score"] = float(getattr(result, "score", 0.0) or 0.0)
            metadata["rerank_score"] = rerank_score
            item.metadata = metadata
            item.score = rerank_score
            reranked.append(item)

        reranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is not None:
            return reranked[:top_k]
        return reranked
