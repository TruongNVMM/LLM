#!/usr/bin/env python3
"""
check_retrieval_quality.py - Automated retrieval checklist for HUST RAG.

This script runs a small suite of Vietnamese queries against the current
retrieval stack and prints:
  - seed hits from the base retriever
  - expanded hits after parent-child expansion
  - a simple verdict based on expected keywords

Usage:
  python -m src.embedding.check_retrieval_quality
  python -m src.embedding.check_retrieval_quality --mode expand
  python -m src.embedding.check_retrieval_quality --query "miễn ngoại ngữ"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable

from src.embedding.retriever import HybridRetriever, RetrievalResult


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_terms: tuple[str, ...] = ()
    note: str = ""


DEFAULT_CASES: list[RetrievalCase] = [
    RetrievalCase(
        query="miễn ngoại ngữ",
        expected_terms=("miễn ngoại ngữ", "miễn học phần ngoại ngữ"),
        note="Routing to attachment / form",
    ),
    RetrievalCase(
        query="đơn xin hoãn thi",
        expected_terms=("hoãn thi", "đơn hoãn thi"),
        note="Form retrieval",
    ),
    RetrievalCase(
        query="rút học phần",
        expected_terms=("rút học phần",),
        note="Body + related form",
    ),
    RetrievalCase(
        query="điều kiện xét học bổng",
        expected_terms=("học bổng", "xét cấp", "điều kiện"),
        note="Legal / policy retrieval",
    ),
    RetrievalCase(
        query="cấp bản sao văn bằng",
        expected_terms=("bản sao văn bằng", "cấp bản sao"),
        note="Long form / legal doc",
    ),
    RetrievalCase(
        query="xác nhận công nợ",
        expected_terms=("công nợ",),
        note="Administrative form",
    ),
    RetrievalCase(
        query="chuyển ngành",
        expected_terms=("chuyển ngành",),
        note="Attachment routing",
    ),
    RetrievalCase(
        query="nghỉ học dài hạn",
        expected_terms=("nghỉ học dài hạn",),
        note="Long-form administrative procedure",
    ),
    RetrievalCase(
        query="thắc mắc điểm",
        expected_terms=("thắc mắc điểm", "phúc tra"),
        note="Specific procedural page",
    ),
    RetrievalCase(
        query="mất thẻ sinh viên",
        expected_terms=("mất thẻ sinh viên",),
        note="Body section retrieval",
    ),
    RetrievalCase(
        query="tại đây",
        expected_terms=(),
        note="Noise / generic-link sanity check",
    ),
]


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _preview(text: str, width: int = 160) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _matches_expected(result: RetrievalResult, expected_terms: Iterable[str]) -> tuple[bool, str]:
    haystacks = [
        result.text,
        result.chunk_id,
        result.metadata.get("attachment_name", ""),
        result.metadata.get("section_header", ""),
        result.metadata.get("source_url", ""),
        result.metadata.get("doc_id", ""),
        result.metadata.get("chunk_type", ""),
    ]
    hay = " \n ".join(_norm(str(v)) for v in haystacks if v is not None)
    for term in expected_terms:
        needle = _norm(term)
        if needle and needle in hay:
            return True, term
    return False, ""


def _format_hit(result: RetrievalResult, rank: int) -> str:
    meta = result.metadata
    chunk_type = str(meta.get("chunk_type", "?"))
    doc_id = str(meta.get("doc_id", "?"))
    section = str(meta.get("section_header", "")).strip() or "-"
    att_name = str(meta.get("attachment_name", "")).strip() or "-"
    att_idx = meta.get("attachment_chunk_index", None)
    attach_info = f"{att_name}"
    if isinstance(att_idx, int) and att_idx >= 0:
        attach_info = f"{att_name}#{att_idx}"
    elif chunk_type == "summary" and att_name != "-":
        attach_info = f"{att_name}#sum"
    match_flag = " "
    return (
        f"  [{rank:>2}] {match_flag} "
        f"{result.chunk_id} | type={chunk_type:<10} | score={result.score:.4f} "
        f"| v={result.vector_rank!s:<3} t={result.text_rank!s:<3} "
        f"| doc={doc_id} | att={attach_info} | sec={section}\n"
        f"       {_preview(result.text)}"
    )


def _rank_verdict(seed: list[RetrievalResult], expanded: list[RetrievalResult], expected_terms: tuple[str, ...]) -> tuple[str, str]:
    if not expected_terms:
        return "MANUAL", "No automatic oracle for this query; inspect the hits."

    def _best_match(results: list[RetrievalResult]) -> tuple[int | None, str]:
        for idx, result in enumerate(results, start=1):
            ok, term = _matches_expected(result, expected_terms)
            if ok:
                return idx, term
        return None, ""

    seed_rank, seed_term = _best_match(seed)
    exp_rank, exp_term = _best_match(expanded)

    if exp_rank == 1:
        return "PASS", f"expanded top-1 matches '{exp_term}'"
    if exp_rank is not None and exp_rank <= 3:
        return "PARTIAL", f"expanded top-{exp_rank} matches '{exp_term}'"
    if seed_rank == 1:
        return "PARTIAL", f"seed top-1 matches '{seed_term}', but expansion did not improve"
    if seed_rank is not None and seed_rank <= 3:
        return "PARTIAL", f"seed top-{seed_rank} matches '{seed_term}', but expansion did not improve"
    return "FAIL", "No expected term found in top results"


def run_case(
    retriever: HybridRetriever,
    case: RetrievalCase,
    *,
    mode: str,
    top_k: int,
    candidate_k: int,
    category_code: str | None,
    body_window: int,
    att_window: int,
    att_short_threshold: int,
) -> tuple[str, str]:
    if mode == "vector":
        seed = retriever.vector_search(case.query, top_k=top_k, category_code=category_code)
        expanded = seed
    elif mode == "text":
        seed = retriever.text_search(case.query, top_k=top_k, category_code=category_code)
        expanded = seed
    elif mode == "hybrid":
        seed = retriever.search(
            case.query,
            top_k=top_k,
            candidate_k=candidate_k,
            category_code=category_code,
        )
        expanded = seed
    else:
        seed = retriever.search(
            case.query,
            top_k=top_k,
            candidate_k=candidate_k,
            category_code=category_code,
        )
        expanded = retriever.expand_with_parent_context(
            seed,
            att_short_threshold=att_short_threshold,
            body_window=body_window,
            att_window=att_window,
        )

    verdict, reason = _rank_verdict(seed, expanded, case.expected_terms)

    print("=" * 92)
    print(f"Query   : {case.query}")
    print(f"Mode    : {mode}")
    if case.note:
        print(f"Note    : {case.note}")
    if case.expected_terms:
        print(f"Expect  : {', '.join(case.expected_terms)}")
    else:
        print("Expect  : (manual review)")

    print(f"\nSeed hits ({len(seed)}):")
    for idx, result in enumerate(seed[:top_k], start=1):
        print(_format_hit(result, idx))

    if mode == "expand":
        print(f"\nExpanded hits ({len(expanded)}):")
        for idx, result in enumerate(expanded[: max(top_k * 2, top_k)], start=1):
            print(_format_hit(result, idx))
    else:
        print(f"\nFinal hits ({len(expanded)}):")
        for idx, result in enumerate(expanded[:top_k], start=1):
            print(_format_hit(result, idx))

    print(f"\nVerdict : {verdict} - {reason}")
    return verdict, reason


def _build_cases(queries: list[str] | None) -> list[RetrievalCase]:
    if not queries:
        return DEFAULT_CASES
    return [RetrievalCase(query=q, note="Custom query") for q in queries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated retrieval checklist for HUST RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["expand", "hybrid", "vector", "text"],
        default="expand",
        help="Retrieval mode to run (default: expand).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results to print.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool for hybrid search before RRF.",
    )
    parser.add_argument(
        "--category",
        choices=["HT", "HC", "HB", "DS", "KN", "HD"],
        default=None,
        help="Optional category filter.",
    )
    parser.add_argument(
        "--body-window",
        type=int,
        default=1,
        help="Neighbor window when expanding body hits.",
    )
    parser.add_argument(
        "--att-window",
        type=int,
        default=1,
        help="Neighbor window when expanding long attachment hits.",
    )
    parser.add_argument(
        "--att-short-threshold",
        type=int,
        default=5,
        help="Attachment chunk count threshold to expand full attachment.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Custom query to run. Can be repeated. If omitted, built-in checklist runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _build_cases(args.query)
    retriever = HybridRetriever(top_k=args.top_k, candidate_k=args.candidate_k)

    print("=" * 92)
    print("HUST RAG retrieval checklist")
    print(f"Cases   : {len(cases)}")
    print(f"Mode    : {args.mode}")
    print(f"Top-k   : {args.top_k}")
    print(f"Cand-k  : {args.candidate_k}")
    if args.category:
        print(f"Category: {args.category}")
    print("=" * 92)

    summary = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "MANUAL": 0}

    for case in cases:
        try:
            verdict, _ = run_case(
                retriever,
                case,
                mode=args.mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                category_code=args.category,
                body_window=args.body_window,
                att_window=args.att_window,
                att_short_threshold=args.att_short_threshold,
            )
        except Exception as exc:
            verdict = "FAIL"
            print("=" * 92)
            print(f"Query   : {case.query}")
            print(f"Mode    : {args.mode}")
            print(f"Verdict : FAIL - retrieval error: {exc}")

        summary[verdict] += 1

    print("=" * 92)
    print(
        "Summary : "
        f"PASS={summary['PASS']} "
        f"PARTIAL={summary['PARTIAL']} "
        f"FAIL={summary['FAIL']} "
        f"MANUAL={summary['MANUAL']}"
    )
    print("=" * 92)
    return 0 if summary["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
