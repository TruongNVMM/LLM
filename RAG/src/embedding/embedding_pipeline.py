#!/usr/bin/env python3
"""
embedding_pipeline.py - Pipeline tạo embedding cho HUST RAG.

Kiến trúc:
  PostgreSQL → đọc chunks chưa embed
  LangChain HuggingFaceEmbeddings → tạo vector (multilingual-e5-large, 1024d)
  Qdrant (Local Mode) → lưu vector + payload
  PostgreSQL → đánh dấu embedded_at

Cách dùng:
  # Embed toàn bộ (incremental - bỏ qua chunks đã embed)
  python -m src.embedding.embedding_pipeline

  # Chỉ embed 50 chunks đầu (kiểm thử)
  python -m src.embedding.embedding_pipeline --limit 50

  # Embed theo category
  python -m src.embedding.embedding_pipeline --category HT

  # Xem thống kê, không embed thực sự
  python -m src.embedding.embedding_pipeline --dry-run

  # Tạo lại collection Qdrant từ đầu (reset)
  python -m src.embedding.embedding_pipeline --recreate-collection
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Đảm bảo project root trong sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


''' Helpers '''
def _detect_device() -> str:
    """Tự động phát hiện GPU/CPU cho embedding model."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS detected")
            return "mps"
    except ImportError:
        pass
    logger.info("Using CPU for embedding")
    return "cpu"


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


''' EmbeddingPipeline '''
class EmbeddingPipeline:
    """
    Pipeline tạo embedding bằng LangChain + Qdrant.

    Flow:
      1. Đọc chunks chưa embed từ PostgreSQL
      2. LangChain HuggingFaceEmbeddings encode text (prefix "passage: ")
      3. Upsert vector vào Qdrant (local mode)
      4. Đánh dấu embedded_at trong PostgreSQL

    Args:
        model_name: HuggingFace model ID. Mặc định: multilingual-e5-large.
        batch_size: Số chunks encode mỗi lần. Mặc định 64 (GPU), giảm nếu OOM.
        device: 'cuda' | 'mps' | 'cpu'. Mặc định tự phát hiện.
        recreate_collection: Xoá và tạo lại Qdrant collection.
    """

    MODEL_NAME     = "intfloat/multilingual-e5-large"
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 64,
        device: str | None = None,
        recreate_collection: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or _detect_device()
        self.recreate_collection = recreate_collection
        self._embeddings = None     # LangChain embedding model (lazy)
        self._qdrant_mgr = None     # QdrantManager (lazy)

    # Lazy-loaded components
    @property
    def embeddings(self):
        """LangChain HuggingFaceEmbeddings wrapper."""
        if self._embeddings is not None:
            return self._embeddings

        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f"Loading LangChain embedding model: {self.model_name} → {self.device}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True,   # cosine sim = dot product
                "batch_size": self.batch_size,
            },
        )
        logger.info("Embedding model loaded.")
        return self._embeddings

    @property
    def qdrant(self):
        """QdrantManager (local mode)."""
        if self._qdrant_mgr is not None:
            return self._qdrant_mgr

        from src.embedding.qdrant_manager import QdrantManager

        self._qdrant_mgr = QdrantManager(recreate=self.recreate_collection)
        return self._qdrant_mgr

    # Encode
    def _encode_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Encode list văn bản.
        Nếu dùng họ model E5, cần thêm prefix "passage: ".
        Các model khác như BAAI/bge-m3 thì không cần prefix.
        """
        is_e5 = "e5" in self.model_name.lower()
        
        if is_e5:
            prepared = [self.PASSAGE_PREFIX + t.strip() for t in texts]
        else:
            prepared = [t.strip() for t in texts]
            
        return self.embeddings.embed_documents(prepared)

    # Main pipeline
    def run(
        self,
        limit: int | None = None,
        category_code: str | None = None,
        dry_run: bool = False,
        commit_every: int = 256,
    ) -> dict:
        """
        Chạy toàn bộ embedding pipeline.

        Args:
            limit: Giới hạn số chunks cần embed (None = toàn bộ).
            category_code: Chỉ embed theo category ('HT', 'HC', ...).
            dry_run: Không ghi dữ liệu, chỉ in thống kê.
            commit_every: Đánh dấu PostgreSQL sau mỗi N chunks (checkpoint).

        Returns:
            dict thống kê: total_embedded, elapsed_sec, chunks_per_sec.
        """
        from src.data_processing.db.connection import (
            get_managed_connection,
            get_unembedded_chunks,
            mark_chunks_as_embedded,
            get_embedding_stats,
        )

        # 1. Thống kê ban đầu
        with get_managed_connection() as conn:
            stats = get_embedding_stats(conn)

        logger.info(
            f"Trạng thái: {stats['embedded']}/{stats['total']} chunks đã embed "
            f"({stats['pct_done']}%)"
        )
        if stats["embedded"] > 0:
            qdrant_stats = self.qdrant.get_stats()
            logger.info(
                f"Qdrant: {qdrant_stats['points_count']} points trong collection "
                f"'{qdrant_stats['collection']}'"
            )

        if stats["unembedded"] == 0:
            logger.info("Tất cả chunks đã được embed. Không có gì cần làm.")
            return stats

        # 2. Lấy chunks chưa embed
        logger.info(
            "Lấy danh sách chunks chưa embed"
            + (f" (limit={limit})" if limit else "")
            + (f" (category={category_code})" if category_code else "")
            + "..."
        )
        with get_managed_connection() as conn:
            chunks = get_unembedded_chunks(conn, limit=limit, category_code=category_code)

        total = len(chunks)
        logger.info(f"Cần embed: {total} chunks")

        if dry_run:
            logger.info("Dry-run: không ghi dữ liệu.")
            return {"dry_run": True, "would_embed": total}

        if total == 0:
            return {"total_embedded": 0, "elapsed_sec": 0}

        # 3. Load model (lần đầu)
        _ = self.embeddings
        _ = self.qdrant

        t0 = time.perf_counter()
        total_embedded = 0
        pending_ids: list[str] = []

        logger.info(
            f"Bắt đầu embed {total} chunks "
            f"(batch_size={self.batch_size}, device={self.device})"
        )

        for batch_num, batch in enumerate(_batched(chunks, self.batch_size), start=1):
            chunk_ids  = [c["id"] for c in batch]
            texts      = [c["text"] for c in batch]
            metadatas  = [c.get("metadata") or {} for c in batch]

            # --- Encode với LangChain ---
            vectors = self._encode_passages(texts)

            # --- Xây dựng payload cho Qdrant ---
            # Lưu các trường cần để filter nhanh trong Qdrant
            qdrant_points = [
                (
                    cid,
                    vec,
                    {
                        "category_code":  meta.get("category_code", ""),
                        "category_name":  meta.get("category_name", ""),
                        "chunk_type":     meta.get("chunk_type", "body"),
                        "is_attachment":  meta.get("is_attachment", False),
                        "source_url":     meta.get("source_url", ""),
                    },
                )
                for cid, vec, meta in zip(chunk_ids, vectors, metadatas)
            ]

            # --- Upsert vào Qdrant ---
            self.qdrant.upsert_points(qdrant_points)
            pending_ids.extend(chunk_ids)
            total_embedded += len(batch)

            # Progress log mỗi 5 batches
            if batch_num % 5 == 0 or total_embedded == total:
                elapsed = time.perf_counter() - t0
                rate = total_embedded / elapsed if elapsed > 0 else 0
                pct  = total_embedded / total * 100
                logger.info(
                    f"  [{pct:5.1f}%] {total_embedded}/{total} chunks"
                    f" | {rate:.1f} chunks/s"
                )

            # --- Checkpoint: đánh dấu PostgreSQL ---
            if len(pending_ids) >= commit_every:
                with get_managed_connection() as conn:
                    n = mark_chunks_as_embedded(conn, pending_ids)
                logger.info(f"Đã cập nhật embedded_at cho {n} chunks trong PostgreSQL")
                pending_ids.clear()

        # Flush remaining ids
        if pending_ids:
            with get_managed_connection() as conn:
                n = mark_chunks_as_embedded(conn, pending_ids)
            logger.info(f"Final flush: {n} chunks cập nhật PostgreSQL")
            pending_ids.clear()

        # 4. Thống kê cuối
        elapsed = time.perf_counter() - t0
        rate = total_embedded / elapsed if elapsed > 0 else 0

        with get_managed_connection() as conn:
            final_stats = get_embedding_stats(conn)
        qdrant_stats = self.qdrant.get_stats()

        logger.info(
            f"\nEmbedding hoàn tất!\n"
            f"   Đã embed       : {total_embedded} chunks\n"
            f"   Thời gian      : {elapsed:.1f}s ({rate:.1f} chunks/s)\n"
            f"   Postgres status: {final_stats['embedded']}/{final_stats['total']} "
            f"({final_stats['pct_done']}%)\n"
            f"   Qdrant points  : {qdrant_stats['points_count']}"
        )

        return {
            "total_embedded": total_embedded,
            "elapsed_sec":    round(elapsed, 2),
            "chunks_per_sec": round(rate, 1),
            **final_stats,
        }



''' CLI '''
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HUST RAG Embedding Pipeline (LangChain + Qdrant)"
    )
    parser.add_argument(
        "--model", default=EmbeddingPipeline.MODEL_NAME,
        help=f"HuggingFace model ID (mặc định: {EmbeddingPipeline.MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Số chunks encode mỗi lần (mặc định: 64, giảm nếu GPU OOM)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số chunks cần embed (mặc định: toàn bộ)",
    )
    parser.add_argument(
        "--category", choices=["HT", "HC", "HB", "DS", "KN", "HD"], default=None,
        help="Chỉ embed theo mã chuyên mục",
    )
    parser.add_argument(
        "--commit-every", type=int, default=256,
        help="Đánh dấu Postgres sau mỗi N chunks (mặc định: 256)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in thống kê, không ghi dữ liệu",
    )
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], default=None,
        help="Device (mặc định: tự phát hiện GPU/CPU)",
    )
    parser.add_argument(
        "--recreate-collection", action="store_true",
        help="Xoá và tạo lại Qdrant collection (⚠️ mất toàn bộ vector đã lưu)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    pipeline = EmbeddingPipeline(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        recreate_collection=args.recreate_collection,
    )
    pipeline.run(
        limit=args.limit,
        category_code=args.category,
        dry_run=args.dry_run,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()
