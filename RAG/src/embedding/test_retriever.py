#!/usr/bin/env python3
"""
Quick test retriever voi model BGE M3.

Chay tu thu muc LLM/RAG:
    python -m src.embedding.test_retriever
    python -m src.embedding.test_retriever "điều kiện xét học bổng" --mode hybrid
    python -m src.embedding.test_retriever "đăng ký học lại" --mode vector --top-k 10

Luu y:
    Vector search chi dung khi Qdrant collection da duoc embed bang cung model
    BAAI/bge-m3. BGE M3 co vector size 1024, tuong thich voi collection hien tai.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_QUERIES = [
    "điều kiện xét học bổng khuyến khích học tập",
    "thủ tục xin hoãn thi",
    "đăng ký học lại học phần",
]


def _print_header(query: str, mode: str, top_k: int, candidate_k: int) -> None:
    print("\n" + "=" * 80)
    print(f"Query      : {query}")
    print(f"Model      : {BGE_M3_MODEL}")
    print(f"Mode       : {mode}")
    print(f"top_k      : {top_k}")
    print(f"candidate_k: {candidate_k}")
    print("=" * 80)


def _print_qdrant_stats() -> None:
    from src.embedding.qdrant_manager import QdrantManager

    try:
        stats = QdrantManager().get_stats()
    except Exception as exc:
        print(f"[WARN] Khong doc duoc Qdrant stats: {exc}")
        return

    print(
        "[Qdrant] "
        f"collection={stats['collection']}, "
        f"points={stats['points_count']}, "
        f"vector_size={stats['vector_size']}, "
        f"path={stats['db_path']}"
    )
    if stats["points_count"] == 0:
        print(
            "[WARN] Collection dang trong. Hay chay embedding pipeline bang "
            "BAAI/bge-m3 truoc khi test vector/hybrid search."
        )


def _print_full_results(results) -> None:
    if not results:
        print("Khong tim thay ket qua phu hop.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}] RRF={r.score:.5f} | v_rank={r.vector_rank} | t_rank={r.text_rank}")
        print(f"Category : {r.category}")
        print(f"Chunk ID : {r.chunk_id}")
        print(f"URL      : {r.source_url}")
        print("Chunk    :")
        print(r.text)

    print(f"\n{'=' * 80}")
    print(f"Tong ket qua: {len(results)}")


def run_query(
    query: str,
    *,
    mode: str,
    top_k: int,
    candidate_k: int,
    category_code: str | None,
    chunk_type: str | None,
) -> None:
    from src.embedding.retriever import HybridRetriever

    retriever = HybridRetriever(
        model_name=BGE_M3_MODEL,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    _print_header(query, mode, top_k, candidate_k)

    if mode == "hybrid":
        results = retriever.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            category_code=category_code,
            chunk_type=chunk_type,
        )
    elif mode == "vector":
        results = retriever.vector_search(
            query,
            top_k=top_k,
            category_code=category_code,
        )
    else:
        results = retriever.text_search(
            query,
            top_k=top_k,
            category_code=category_code,
        )

    _print_full_results(results)


def _iter_queries(query: str | None, use_defaults: bool) -> Iterable[str]:
    if query:
        yield query
        return
    if use_defaults:
        yield from DEFAULT_QUERIES
        return
    yield DEFAULT_QUERIES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test retrieve HUST RAG bang embedding model BGE M3."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Cau hoi can retrieve. Bo trong de dung query mac dinh.",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vector", "text"],
        default="hybrid",
        help="Che do retrieve.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--category",
        dest="category_code",
        choices=["HT", "HC", "HB", "DS", "KN", "HD"],
        default=None,
        help="Loc theo ma chuyen muc.",
    )
    parser.add_argument(
        "--chunk-type",
        choices=["body", "attachment"],
        default=None,
        help="Loc theo loai chunk khi chay hybrid search.",
    )
    parser.add_argument(
        "--all-defaults",
        action="store_true",
        help="Chay tat ca query mac dinh neu khong truyen query.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Bat logging INFO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _print_qdrant_stats()

    for query in _iter_queries(args.query, args.all_defaults):
        run_query(
            query,
            mode=args.mode,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            category_code=args.category_code,
            chunk_type=args.chunk_type,
        )


if __name__ == "__main__":
    main()
