"""
qdrant_manager.py - Quản lý Qdrant collection cho HUST RAG Pipeline.

Qdrant chạy ở LOCAL MODE (không cần Docker/Server):
  - Lưu dữ liệu vào thư mục cục bộ: data/qdrant_db/
  - Hoàn toàn tương thích Windows
  - Tự động tạo collection nếu chưa tồn tại

Cấu hình Collection:
  - Tên: "hust_rag_chunks"
  - Vector size: 1024 (multilingual-e5-large)
  - Distance: Cosine
  - Payload lưu: chunk_id, category_code, chunk_type, is_attachment

Cách dùng:
  from src.embedding.qdrant_manager import QdrantManager

  manager = QdrantManager()
  manager.upsert_points([(chunk_id, vector, payload), ...])
  results = manager.search(query_vector, top_k=20)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


''' Config '''
COLLECTION_NAME = "hust_rag_chunks"
VECTOR_SIZE     = 1024          # multilingual-e5-large
QDRANT_DB_PATH  = str(Path(__file__).parent.parent.parent / "data" / "qdrant_db")


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Chuyển chunk_id dạng chuỗi → UUID dạng chuỗi (UUIDv5).
    Qdrant yêu cầu ID là UUID hoặc số nguyên unsigned 64-bit.
    UUIDv5 đảm bảo cùng chunk_id luôn cho cùng UUID (deterministic).
    """
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL
    return str(uuid.uuid5(namespace, chunk_id))


''' QdrantManager '''
class QdrantManager:
    """
    Lớp quản lý Qdrant collection chạy ở Local Mode.

    Args:
        db_path: Thư mục lưu dữ liệu Qdrant (mặc định: data/qdrant_db/).
        collection_name: Tên collection (mặc định: hust_rag_chunks).
        vector_size: Số chiều vector (mặc định: 1024).
        recreate: Xoá và tạo lại collection nếu True.
    """

    def __init__(
        self,
        db_path: str = QDRANT_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
        recreate: bool = False,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None
        self._ensure_collection(recreate=recreate)

    # Client & Collection
    def _get_client(self):
        """Lấy hoặc tạo Qdrant client (lazy init)."""
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient

        os.makedirs(self.db_path, exist_ok=True)
        logger.info(f"Khởi tạo Qdrant local tại: {self.db_path}")
        self._client = QdrantClient(path=self.db_path)
        return self._client

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Tạo collection nếu chưa tồn tại, hoặc recreate nếu được yêu cầu."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams,
            HnswConfigDiff, OptimizersConfigDiff,
        )

        client = self._get_client()
        exists = client.collection_exists(self.collection_name)

        if exists and recreate:
            logger.warning(f"Xoá collection '{self.collection_name}' để tạo lại...")
            client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Tạo Qdrant collection: '{self.collection_name}' "
                        f"(size={self.vector_size}, distance=Cosine)")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,               # bi-directional links – cân bằng tốc độ/recall
                    ef_construct=100,   # độ sâu khi xây dựng index
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' đã được tạo.")
        else:
            info = client.get_collection(self.collection_name)
            count = info.points_count or 0
            logger.info(
                f"Collection '{self.collection_name}' đã tồn tại "
                f"({count} points)."
            )

    # Upsert
    def upsert_points(
        self,
        points: list[tuple[str, list[float], dict[str, Any]]],
        batch_size: int = 128,
    ) -> int:
        """
        Upsert danh sách (chunk_id, vector, payload) vào Qdrant.

        Args:
            points: List of (chunk_id, vector, payload).
                - chunk_id: str – ID gốc của chunk (sẽ được hash → UUID)
                - vector: list[float] – vector 1024d đã normalize
                - payload: dict – metadata lưu kèm (category_code, chunk_type...)
            batch_size: Số points upsert mỗi lần (mặc định: 128).

        Returns:
            Số points đã upsert thành công.
        """
        from qdrant_client.models import PointStruct

        client = self._get_client()
        total = 0

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant_points = [
                PointStruct(
                    id=_chunk_id_to_uuid(chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,   # lưu ID gốc vào payload để map ngược
                        **payload,
                    },
                )
                for chunk_id, vector, payload in batch
            ]
            client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
            )
            total += len(batch)

        return total

    # Search
    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        category_code: str | None = None,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """
        Tìm kiếm vector similarity trong Qdrant.

        Args:
            query_vector: Query vector đã normalize (1024d, prefix "query: ").
            top_k: Số kết quả trả về.
            category_code: Lọc theo mã chuyên mục ('HT', 'HC', ...).
            chunk_type: Lọc theo loại chunk ('body' | 'attachment' | None).

        Returns:
            List[dict] với các key: chunk_id, score, payload.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, AndCondition

        # Xây dựng filter payload
        filter_conditions = []
        if category_code:
            filter_conditions.append(
                FieldCondition(key="category_code", match=MatchValue(value=category_code))
            )
        if chunk_type:
            filter_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))
            )

        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        client = self._get_client()
        results = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id", ""),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    # Stats
    def get_stats(self) -> dict:
        """Trả về thống kê collection."""
        client = self._get_client()
        info = client.get_collection(self.collection_name)
        return {
            "collection": self.collection_name,
            "points_count": info.points_count or 0,
            "vector_size": self.vector_size,
            "db_path": self.db_path,
        }

    def delete_collection(self) -> None:
        """Xoá hoàn toàn collection (cẩn thận!)."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        logger.warning(f"Đã xoá collection '{self.collection_name}'.")
