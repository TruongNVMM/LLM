#!/usr/bin/env python3
"""
retriever.py - Hybrid Search retriever cho HUST RAG Pipeline.

Kiến trúc: Qdrant (vector) + PostgreSQL (full-text) → RRF merge

Chiến lược Hybrid Search (Reciprocal Rank Fusion):
  1. Qdrant Vector Search  → Top-K candidates (ngữ nghĩa)
  2. Postgres Full-Text    → Top-K candidates (từ khoá)
  3. RRF Merge             → Trộn kết quả dựa trên thứ hạng
  4. Fetch full text       → Lấy nội dung đầy đủ từ Postgres

RRF Score công thức:
    score = Σ [ 1 / (k + rank_i) ]
    Với k=60 (hằng số chuẩn của RRF), rank_i là thứ hạng trong từng danh sách.

LangChain Integration:
  - Dùng LangChain HuggingFaceEmbeddings để encode query (prefix "query: ")
  - Dùng LangChain QdrantVectorStore để search vector
  - Kết quả cuối được trả về dưới dạng LangChain Document

Cách dùng:
  from src.embedding.retriever import HybridRetriever

  retriever = HybridRetriever()

  # Hybrid search (Qdrant + Postgres + RRF)
  docs = retriever.invoke("đăng ký học lại")

  # Search nâng cao với filter
  docs = retriever.search("học bổng", top_k=5, category_code="HB")

  # Pure vector search (chỉ Qdrant)
  docs = retriever.vector_search("xét tốt nghiệp", top_k=10)

  # Pure text search (chỉ Postgres)
  docs = retriever.text_search("ký túc xá", top_k=10)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)

# Đảm bảo project root trong sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


''' RRF helper '''
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: trộn nhiều danh sách xếp hạng.

    Args:
        ranked_lists: Danh sách các list chunk_id đã sắp xếp (tốt nhất đầu tiên).
        k: Hằng số RRF (mặc định 60 theo paper gốc).

    Returns:
        List[(chunk_id, rrf_score)] sắp xếp giảm dần theo score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


''' RetrievalResult dataclass '''
@dataclass
class RetrievalResult:
    """Kết quả trả về cho một chunk."""
    chunk_id:     str
    text:         str
    score:        float            # RRF score (hoặc cosine/text score nếu single-mode)
    vector_rank:  int | None       # Thứ hạng trong Qdrant result (None nếu không có)
    text_rank:    int | None       # Thứ hạng trong Postgres result (None nếu không có)
    metadata:     dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return self.metadata.get("category_name", "")

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def to_document(self) -> Document:
        """Chuyển sang LangChain Document."""
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id":   self.chunk_id,
                "score":      self.score,
                "vector_rank": self.vector_rank,
                "text_rank":  self.text_rank,
                **self.metadata,
            },
        )

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"v_rank={self.vector_rank}, t_rank={self.text_rank}, "
            f"category={self.category!r})"
        )


''' HybridRetriever – implements LangChain BaseRetriever '''
class HybridRetriever(BaseRetriever):
    """
    Hybrid Search retriever kết hợp Qdrant + PostgreSQL với RRF.

    Implement LangChain BaseRetriever để tích hợp trực tiếp vào
    LangChain chains (RAG chain, ConversationalRetrievalChain...).

    Args:
        model_name: HuggingFace model ID cho embedding query.
        top_k: Số kết quả trả về cuối cùng.
        candidate_k: Số candidates lấy từ mỗi engine trước khi RRF.
        rrf_k: Hằng số RRF (mặc định 60).
        device: 'cuda' | 'mps' | 'cpu' | None.

    Ví dụ tích hợp LangChain chain:
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        retriever = HybridRetriever(top_k=5)
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever,
        )
        answer = chain.invoke("Điều kiện xét học bổng là gì?")
    """

    model_name:  str = "intfloat/multilingual-e5-large"
    top_k:       int = 5
    candidate_k: int = 20
    rrf_k:       int = 60

    # Các thuộc tính private (Pydantic v1 trong LangChain dùng _ prefix)
    class Config:
        arbitrary_types_allowed = True

    # Internal state (không dùng Pydantic field để tránh serialize)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store private state as object attributes
        object.__setattr__(self, '_embeddings_model', None)
        object.__setattr__(self, '_qdrant_mgr', None)
        object.__setattr__(self, '_device', None)

    @property
    def _embed_model(self):
        if object.__getattribute__(self, '_embeddings_model') is not None:
            return object.__getattribute__(self, '_embeddings_model')

        from langchain_huggingface import HuggingFaceEmbeddings
        import torch

        device = object.__getattribute__(self, '_device')
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            object.__setattr__(self, '_device', device)

        logger.info(f"Loading embedding model: {self.model_name} → {device}")
        model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        object.__setattr__(self, '_embeddings_model', model)
        return model

    @property
    def _qdrant(self):
        if object.__getattribute__(self, '_qdrant_mgr') is not None:
            return object.__getattribute__(self, '_qdrant_mgr')

        from src.embedding.qdrant_manager import QdrantManager
        mgr = QdrantManager()
        object.__setattr__(self, '_qdrant_mgr', mgr)
        return mgr

    # Encode query
    def _encode_query(self, query: str) -> list[float]:
        """
        Encode câu hỏi → vector 1024d.
        Nếu dùng model E5, cần thêm prefix "query: ".
        Các model khác như BAAI/bge-m3 thì không cần prefix.
        """
        is_e5 = "e5" in self.model_name.lower()
        
        if is_e5:
            prepared = "query: " + query.strip()
        else:
            prepared = query.strip()
            
        return self._embed_model.embed_query(prepared)

    # Individual search methods
    def _vector_search_ids(
        self,
        query_vector: list[float],
        candidate_k: int,
        category_code: str | None,
        chunk_type: str | None,
    ) -> list[str]:
        """Tìm kiếm Qdrant → trả về danh sách chunk_id (đã sắp xếp)."""
        hits = self._qdrant.search(
            query_vector=query_vector,
            top_k=candidate_k,
            category_code=category_code,
            chunk_type=chunk_type,
        )
        return [h["chunk_id"] for h in hits]

    def _text_search_ids(
        self,
        query_text: str,
        candidate_k: int,
        category_code: str | None,
        chunk_type: str | None,
    ) -> list[str]:
        """Full-text search PostgreSQL → trả về danh sách chunk_id (đã sắp xếp)."""
        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        conditions = ["embedded_at IS NOT NULL"]
        params: dict = {
            "query_text": query_text,
            "limit": candidate_k,
        }

        if category_code:
            conditions.append("metadata->>'category_code' = %(category_code)s")
            params["category_code"] = category_code

        if chunk_type:
            conditions.append("chunk_type = %(chunk_type)s")
            params["chunk_type"] = chunk_type

        where = " AND ".join(conditions)

        sql = f"""
            SELECT id,
                   ts_rank(
                       to_tsvector('simple', text),
                       plainto_tsquery('simple', %(query_text)s)
                   ) AS rank
            FROM rag_chunks
            WHERE {where}
              AND plainto_tsquery('simple', %(query_text)s) @@ to_tsvector('simple', text)
            ORDER BY rank DESC
            LIMIT %(limit)s;
        """
        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [row["id"] for row in cur.fetchall()]

    def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Lấy nội dung đầy đủ từ PostgreSQL theo danh sách chunk_id."""
        if not chunk_ids:
            return {}

        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        sql = """
            SELECT id, text, metadata
            FROM rag_chunks
            WHERE id = ANY(%(ids)s);
        """
        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, {"ids": chunk_ids})
                rows = cur.fetchall()

        return {row["id"]: dict(row) for row in rows}

    # Public search methods
    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        category_code: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid Search = Qdrant Vector + Postgres Full-Text + RRF Merge.

        Args:
            query: Câu hỏi người dùng (tiếng Việt).
            top_k: Số kết quả cuối trả về.
            candidate_k: Số candidates lấy từ mỗi engine.
            category_code: Lọc theo chuyên mục ('HT','HC','HB','DS','KN','HD').
            chunk_type: Lọc theo loại chunk ('body' | 'attachment' | None).

        Returns:
            List[RetrievalResult] sắp xếp theo RRF score giảm dần.
        """
        k       = top_k       or self.top_k
        cand_k  = candidate_k or self.candidate_k

        # 1. Encode query
        query_vec = self._encode_query(query)

        # 2. Tìm kiếm song song từ 2 nguồn
        vector_ids = self._vector_search_ids(query_vec, cand_k, category_code, chunk_type)
        text_ids   = self._text_search_ids(query, cand_k, category_code, chunk_type)

        logger.debug(
            f"Vector search: {len(vector_ids)} results, "
            f"Text search: {len(text_ids)} results"
        )

        # 3. RRF merge
        rrf_results = reciprocal_rank_fusion(
            [vector_ids, text_ids], k=self.rrf_k
        )
        top_ids = [cid for cid, _ in rrf_results[:k]]
        rrf_score_map = {cid: score for cid, score in rrf_results}

        # 4. Lấy nội dung đầy đủ từ Postgres
        chunks_data = self._fetch_chunks_by_ids(top_ids)

        # Xây dựng reverse rank maps để debug
        v_rank_map = {cid: r + 1 for r, cid in enumerate(vector_ids)}
        t_rank_map = {cid: r + 1 for r, cid in enumerate(text_ids)}

        results = []
        for cid in top_ids:
            data = chunks_data.get(cid)
            if not data:
                continue
            results.append(RetrievalResult(
                chunk_id    = cid,
                text        = data["text"],
                score       = rrf_score_map.get(cid, 0.0),
                vector_rank = v_rank_map.get(cid),
                text_rank   = t_rank_map.get(cid),
                metadata    = dict(data.get("metadata") or {}),
            ))

        return results

    def vector_search(
        self,
        query: str,
        top_k: int | None = None,
        category_code: str | None = None,
    ) -> list[RetrievalResult]:
        """Pure Vector Search (chỉ Qdrant). Nhanh hơn hybrid."""
        k = top_k or self.top_k
        query_vec = self._encode_query(query)
        hits = self._qdrant.search(
            query_vector=query_vec,
            top_k=k,
            category_code=category_code,
        )
        ids = [h["chunk_id"] for h in hits]
        score_map = {h["chunk_id"]: h["score"] for h in hits}
        chunks_data = self._fetch_chunks_by_ids(ids)

        return [
            RetrievalResult(
                chunk_id    = cid,
                text        = chunks_data[cid]["text"],
                score       = score_map.get(cid, 0.0),
                vector_rank = r + 1,
                text_rank   = None,
                metadata    = dict(chunks_data[cid].get("metadata") or {}),
            )
            for r, cid in enumerate(ids)
            if cid in chunks_data
        ]

    def text_search(
        self,
        query: str,
        top_k: int | None = None,
        category_code: str | None = None,
    ) -> list[RetrievalResult]:
        """Pure Full-Text Search (႕ỉ PostgreSQL)."""
        k = top_k or self.top_k
        ids = self._text_search_ids(query, k, category_code, chunk_type=None)
        chunks_data = self._fetch_chunks_by_ids(ids)

        return [
            RetrievalResult(
                chunk_id    = cid,
                text        = chunks_data[cid]["text"],
                score       = 1.0 / (self.rrf_k + r + 1),
                vector_rank = None,
                text_rank   = r + 1,
                metadata    = dict(chunks_data[cid].get("metadata") or {}),
            )
            for r, cid in enumerate(ids)
            if cid in chunks_data
        ]

    # ── Parent-child retrieval helpers ────────────────────────────────────────

    def _fetch_neighbor_chunks(
        self,
        parent_id: str,
        chunk_index: int,
        window: int = 1,
        exclude_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Lấy tối đa 'window' chunks trước và sau chunk_index cùng parent_id.
        Bỏ qua summary chunks (chỉ lấy body và attachment).

        Args:
            parent_id: ID bài viết cha (ví dụ: 'hust_sotay_69').
            chunk_index: Index của hit chunk.
            window: Số chunk lấy thêm mỗi phía (±1 mặc định).
            exclude_types: Loại chunk cần bỏ qua (mặc định: ['summary']).
        """
        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        excl = exclude_types or ['summary']
        excl_sql = ', '.join(f"'{t}'" for t in excl)

        sql = f"""
            SELECT id, text, metadata,
                   (metadata->>'chunk_index')::int AS cidx
            FROM rag_chunks
            WHERE metadata->>'doc_id' = %(parent_id)s::text
              AND (metadata->>'chunk_index')::int
                  BETWEEN %(lo)s AND %(hi)s
              AND chunk_type NOT IN ({excl_sql})
              AND embedded_at IS NOT NULL
            ORDER BY cidx;
        """
        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, {
                    'parent_id': parent_id,
                    'lo': chunk_index - window,
                    'hi': chunk_index + window,
                })
                return [dict(row) for row in cur.fetchall()]

    def _fetch_attachment_chunks(
        self,
        parent_id: str,
        attachment_name: str,
        chunk_index: int | None = None,
        window: int | None = None,
    ) -> list[dict]:
        """
        Lấy chunks của một attachment.

        - Nếu window=None: lấy toàn bộ attachment (dùng khi attachment ngắn).
        - Nếu window có giá trị: lấy ±window xung quanh chunk_index (attachment dài).
        """
        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        if window is not None and chunk_index is not None:
            sql = """
                SELECT id, text, metadata,
                       (metadata->>'chunk_index')::int AS cidx
                FROM rag_chunks
                WHERE metadata->>'doc_id' = %(parent_id)s::text
                  AND metadata->>'attachment_name' = %(att_name)s::text
                  AND chunk_type = 'attachment'
                  AND (metadata->>'chunk_index')::int
                      BETWEEN %(lo)s AND %(hi)s
                  AND embedded_at IS NOT NULL
                ORDER BY cidx;
            """
            params = {
                'parent_id': parent_id,
                'att_name':  attachment_name,
                'lo': chunk_index - window,
                'hi': chunk_index + window,
            }
        else:
            sql = """
                SELECT id, text, metadata,
                       (metadata->>'chunk_index')::int AS cidx
                FROM rag_chunks
                WHERE metadata->>'doc_id' = %(parent_id)s::text
                  AND metadata->>'attachment_name' = %(att_name)s::text
                  AND chunk_type = 'attachment'
                  AND embedded_at IS NOT NULL
                ORDER BY cidx;
            """
            params = {'parent_id': parent_id, 'att_name': attachment_name}

        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def _count_attachment_chunks(
        self,
        parent_id: str,
        attachment_name: str,
    ) -> int:
        """Dem số content chunks của một attachment."""
        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        sql = """
            SELECT COUNT(*) AS cnt
            FROM rag_chunks
            WHERE metadata->>'doc_id' = %(parent_id)s::text
              AND metadata->>'attachment_name' = %(att_name)s::text
              AND chunk_type = 'attachment'
              AND embedded_at IS NOT NULL;
        """
        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, {'parent_id': parent_id, 'att_name': attachment_name})
                return cur.fetchone()['cnt']

    def _fetch_parent_content_chunks(
        self,
        parent_id: str,
    ) -> list[dict]:
        """Lấy tất cả content chunks (body + attachment) của một bài (dùng khi hit summary chunk)."""
        import psycopg2.extras
        from src.data_processing.db.connection import get_managed_connection

        sql = """
            SELECT id, text, metadata,
                   (metadata->>'chunk_index')::int AS cidx
            FROM rag_chunks
            WHERE metadata->>'doc_id' = %(parent_id)s::text
              AND chunk_type IN ('body', 'attachment')
              AND embedded_at IS NOT NULL
            ORDER BY cidx
            LIMIT 10;
        """
        with get_managed_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, {'parent_id': parent_id})
                return [dict(row) for row in cur.fetchall()]

    def _rows_to_results(
        self,
        rows: list[dict],
        base_score: float = 0.0,
        source: str = 'expand',
    ) -> list[RetrievalResult]:
        """Chuyển DB rows thành RetrievalResult."""
        results = []
        for row in rows:
            results.append(RetrievalResult(
                chunk_id    = row['id'],
                text        = row['text'],
                score       = base_score,
                vector_rank = None,
                text_rank   = None,
                metadata    = dict(row.get('metadata') or {}),
            ))
        return results

    def expand_with_parent_context(
        self,
        results: list[RetrievalResult],
        att_short_threshold: int = 5,
        body_window: int = 1,
        att_window: int = 1,
    ) -> list[RetrievalResult]:
        """
        Parent-child retrieval expansion (Strategy 5).

        Rules:
          - body chunk hit    : lấy ±body_window chunks cùng doc_id
          - attachment hit    : nếu attachment ≤ att_short_threshold chunks
                                  → lấy toàn bộ attachment
                                nếu dài → lấy ±att_window chunk cùng attachment_name
          - summary hit       : expand sang các content chunks cùng parent_id

        Kết quả: deduplicate + sort theo chunk_index để LLM thấy nội dung liền mạch.

        Args:
            results: Danh sách hit từ search().
            att_short_threshold: Số chunks để cói attachment là 'ngắn' (lấy toàn bộ).
            body_window: Số chunk lấy thêm mỗi phía cho body chunk.
            att_window: Số chunk lấy thêm mỗi phía cho attachment chunk dài.

        Returns:
            List[RetrievalResult] deduplicated + sorted by chunk_index.
        """
        seen_ids: set[str] = set()
        expanded: list[RetrievalResult] = []

        for hit in results:
            chunk_type  = hit.metadata.get('chunk_type', 'body')
            parent_id   = hit.metadata.get('doc_id', '')
            chunk_index = hit.metadata.get('chunk_index', 0)
            att_name    = hit.metadata.get('attachment_name', '')

            # --- Add the hit itself first ---
            if hit.chunk_id not in seen_ids:
                expanded.append(hit)
                seen_ids.add(hit.chunk_id)

            try:
                if chunk_type == 'body':
                    # Lấy ±1 body chunk cùng doc
                    rows = self._fetch_neighbor_chunks(
                        parent_id, chunk_index, window=body_window
                    )
                    for r in self._rows_to_results(rows, base_score=hit.score * 0.8):
                        if r.chunk_id not in seen_ids:
                            expanded.append(r)
                            seen_ids.add(r.chunk_id)

                elif chunk_type == 'attachment' and att_name and parent_id:
                    # Kiểm tra attachment dài hay ngắn
                    total = self._count_attachment_chunks(parent_id, att_name)
                    if total <= att_short_threshold:
                        # Ngắn → lấy toàn bộ
                        rows = self._fetch_attachment_chunks(parent_id, att_name)
                    else:
                        # Dài → lấy ±att_window
                        rows = self._fetch_attachment_chunks(
                            parent_id, att_name, chunk_index=chunk_index, window=att_window
                        )
                    for r in self._rows_to_results(rows, base_score=hit.score * 0.8):
                        if r.chunk_id not in seen_ids:
                            expanded.append(r)
                            seen_ids.add(r.chunk_id)

                elif chunk_type == 'summary':
                    # Summary hit → expand sang content chunks cùng parent
                    if att_name:
                        # Summary của attachment → lấy toàn bộ attachment
                        rows = self._fetch_attachment_chunks(parent_id, att_name)
                    else:
                        # Summary của bài viết → lấy các body chunks đầu tiên
                        rows = self._fetch_parent_content_chunks(parent_id)
                    for r in self._rows_to_results(rows, base_score=hit.score * 0.9):
                        if r.chunk_id not in seen_ids:
                            expanded.append(r)
                            seen_ids.add(r.chunk_id)

            except Exception as e:
                logger.warning(f"expand_with_parent_context failed for {hit.chunk_id}: {e}")
                continue

        # Deduplicate + sort by chunk_index để LLM thấy nội dung liền mạch
        def _sort_key(r: RetrievalResult) -> tuple:
            idx = r.metadata.get('chunk_index', 9999)
            doc = r.metadata.get('doc_id', '')
            return (doc, idx)

        expanded.sort(key=_sort_key)
        return expanded

    def search_with_expansion(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        category_code: str | None = None,
        att_short_threshold: int = 5,
        body_window: int = 1,
        att_window: int = 1,
    ) -> list[RetrievalResult]:
        """
        Hybrid Search + Parent-Child Expansion.

        Bước 1: Hybrid search (Qdrant + Postgres + RRF) trên tất cả chunk types.
        Bước 2: Với mỗi hit, expand ra parent context theo rule:
               - body   → ±body_window chunks cùng doc
               - attach → toàn bộ attachment (nếu ngắn) hoặc ±att_window
               - summary→ các content chunks cùng parent
        Bước 3: Deduplicate + sort theo chunk_index.

        Args:
            query: Câu hỏi người dùng.
            top_k: Số hit ban đầu cần lấy.
            candidate_k: Số candidates cho mỗi search engine.
            category_code: Lọc theo chuyên mục.
            att_short_threshold: attachment ≤ N chunks → lấy toàn bộ.
            body_window: Neighbors cho body chunk.
            att_window: Neighbors cho attachment chunk dài.

        Returns:
            List[RetrievalResult] đã expand và sắp xếp.
        """
        hits = self.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            category_code=category_code,
        )
        return self.expand_with_parent_context(
            hits,
            att_short_threshold=att_short_threshold,
            body_window=body_window,
            att_window=att_window,
        )


    # LangChain BaseRetriever interface
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Implement LangChain BaseRetriever.
        Được gọi khi dùng retriever trong LangChain chain.

        Ví dụ:
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            chain.invoke("câu hỏi")
        """
        results = self.search(query)
        return [r.to_document() for r in results]

    # Utility
    def print_results(self, results: list[RetrievalResult]) -> None:
        """In kết quả theo format dễ đọc."""
        if not results:
            print("Không tìm thấy kết quả phù hợp.")
            return
        for i, r in enumerate(results, 1):
            print(f"\n{'='*65}")
            print(
                f"[{i}] RRF={r.score:.5f}  "
                f"| v_rank={r.vector_rank}  t_rank={r.text_rank}"
            )
            print(f"     Category : {r.category}")
            print(f"     Chunk ID : {r.chunk_id}")
            print(f"     URL      : {r.source_url}")
            preview = r.text[:350].replace("\n", " ")
            print(f"     Preview  : {preview}{'...' if len(r.text) > 350 else ''}")
        print(f"\n{'='*65}")
        print(f"Tổng kết quả: {len(results)}")


''' CLI / Quick test '''
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test HUST RAG Hybrid Retriever (Qdrant + Postgres + RRF)"
    )
    parser.add_argument(
        "query", nargs="?", default="điều kiện đăng ký học lại",
        help="Câu hỏi cần tìm kiếm"
    )
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--category", choices=["HT", "HC", "HB", "DS", "KN", "HD"], default=None
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "text", "expand"], default="hybrid",
        help="Chế độ tìm kiếm (expand = hybrid + parent-child expansion)"
    )
    args = parser.parse_args()

    retriever = HybridRetriever(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print(f"\nTìm kiếm: {args.query!r}  (mode={args.mode}, top_k={args.top_k})")

    if args.mode == "hybrid":
        results = retriever.search(args.query, category_code=args.category)
    elif args.mode == "vector":
        results = retriever.vector_search(args.query, category_code=args.category)
    elif args.mode == "text":
        results = retriever.text_search(args.query, category_code=args.category)
    else:  # expand
        results = retriever.search_with_expansion(
            args.query, top_k=args.top_k, category_code=args.category
        )

    if not results:
        print("Không tìm thấy kết quả phù hợp.")
    else:
        for i, r in enumerate(results, 1):
            chunk_type = r.metadata.get('chunk_type', '?')
            print(f"\n{'='*65}")
            print(
                f"[{i}] RRF={r.score:.5f}  "
                f"| type={chunk_type:10s} "
                f"| v_rank={r.vector_rank}  t_rank={r.text_rank}"
            )
            print(f"     Category : {r.category}")
            print(f"     Chunk ID : {r.chunk_id}")
            print(f"     URL      : {r.source_url}")
            preview = r.text[:350].replace("\n", " ")
            print(f"     Preview  : {preview}{'...' if len(r.text) > 350 else ''}")
        print(f"\n{'='*65}")
        print(f"Tổng kết quả: {len(results)}")
