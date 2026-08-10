"""
connection.py – Quản lý kết nối PostgreSQL cho HUST RAG pipeline.

Config được load từ biến môi trường hoặc file .env trong thư mục gốc dự án.
Thứ tự ưu tiên: biến môi trường hệ thống > .env file.

Biến môi trường cần thiết:
    POSTGRES_HOST      (mặc định: localhost)
    POSTGRES_PORT      (mặc định: 5432)
    POSTGRES_DB        (mặc định: hust_rag)
    POSTGRES_USER      (mặc định: postgres)
    POSTGRES_PASSWORD  (bắt buộc)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

''' Tìm và load .env file '''
def _load_dotenv() -> None:
    """Load .env file từ thư mục gốc dự án (đi lên từ file này)."""
    # Tìm .env từ thư mục gốc dự án: src/data_processing/db/ → 3 cấp lên
    search_dirs = [
        Path(__file__).parent,          # db/
        Path(__file__).parent.parent,   # data_processing/
        Path(__file__).parent.parent.parent,  # src/
        Path(__file__).parent.parent.parent.parent,  # RAG/ (project root)
        Path.cwd(),                     # Working directory
    ]
    for directory in search_dirs:
        env_file = directory / ".env"
        if env_file.exists():
            logger.debug(f"Loading .env from {env_file}")
            _parse_dotenv(env_file)
            return
    logger.debug("No .env file found, using system environment variables only.")


def _parse_dotenv(path: Path) -> None:
    """Parse đơn giản file .env (không cần thư viện python-dotenv)."""
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Không ghi đè biến môi trường đã tồn tại
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as e:
        logger.warning(f"Could not read .env file: {e}")


# Load .env khi module được import
_load_dotenv()


''' Config '''
def get_db_config() -> dict[str, str | int]:
    """Đọc config kết nối PostgreSQL từ biến môi trường."""
    password = os.environ.get("POSTGRES_PASSWORD", "")
    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD chưa được thiết lập. "
            "Hãy tạo file .env trong thư mục gốc dự án với nội dung:\n"
            "  POSTGRES_PASSWORD=your_password_here\n"
            "Xem .env.example để biết thêm."
        )
    return {
        "host":     os.environ.get("POSTGRES_HOST", "localhost"),
        "port":     int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname":   os.environ.get("POSTGRES_DB", "hust_rag"),
        "user":     os.environ.get("POSTGRES_USER", "postgres"),
        "password": password,
    }


''' Kết nối '''
def get_connection() -> psycopg2.extensions.connection:
    """
    Tạo và trả về một connection PostgreSQL mới.
    
    Caller có trách nhiệm đóng connection sau khi dùng xong,
    hoặc dùng `get_managed_connection()` context manager.
    """
    config = get_db_config()
    try:
        conn = psycopg2.connect(
            **config,
            connect_timeout=10,
            options="-c client_encoding=UTF8",
        )
        conn.autocommit = False
        logger.debug(
            f"Connected to PostgreSQL: {config['user']}@{config['host']}:{config['port']}/{config['dbname']}"
        )
        return conn
    except psycopg2.OperationalError as e:
        raise ConnectionError(
            f"Không thể kết nối PostgreSQL tại "
            f"{config['host']}:{config['port']}/{config['dbname']}: {e}"
        ) from e


@contextmanager
def get_managed_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager trả về connection PostgreSQL.
    Tự động commit nếu không có lỗi, rollback nếu có exception,
    và luôn đóng connection khi kết thúc.

    Dùng:
        with get_managed_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


''' Khởi tạo schema '''
def init_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Chạy schema.sql để tạo tất cả bảng và index (nếu chưa tồn tại).
    Idempotent – an toàn để gọi nhiều lần.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Không tìm thấy schema.sql tại: {schema_path}")
    
    sql = schema_path.read_text(encoding="utf-8")
    
    # Bỏ qua lệnh tạo pg_trgm index nếu extension chưa được cài
    # (sẽ tạo lại sau khi cài extension)
    try:
        with conn.cursor() as cur:
            # Thử enable pg_trgm extension (cần superuser, bỏ qua nếu lỗi)
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                conn.commit()
                logger.info("pg_trgm extension đã sẵn sàng.")
            except psycopg2.Error:
                conn.rollback()
                logger.warning(
                    "Không thể tạo pg_trgm extension (cần quyền superuser). "
                    "GIN full-text index sẽ bị bỏ qua."
                )
                # Bỏ lệnh tạo GIN trgm index
                sql = "\n".join(
                    line for line in sql.splitlines()
                    if "gin_trgm_ops" not in line
                )
            
            cur.execute(sql)
            conn.commit()
            logger.info("Schema khởi tạo thành công.")
    except psycopg2.Error as e:
        conn.rollback()
        raise RuntimeError(f"Lỗi khi khởi tạo schema: {e}") from e


''' Upsert helpers '''
def upsert_article(conn: psycopg2.extensions.connection, article: dict) -> None:
    """
    Insert hoặc update một bài viết vào bảng `articles`.
    Dùng ON CONFLICT (doc_id) DO UPDATE để hỗ trợ incremental crawl.
    """
    sql = """
        INSERT INTO articles (
            doc_id, title, type_doc,
            category_code, category_name, category_desc, category_order,
            source_url, time_create, status, creator_id,
            html, text, rag_text,
            links, attachments, keywords, crawled_at
        ) VALUES (
            %(doc_id)s, %(title)s, %(type_doc)s,
            %(category_code)s, %(category_name)s, %(category_desc)s, %(category_order)s,
            %(source_url)s, %(time_create)s, %(status)s, %(creator_id)s,
            %(html)s, %(text)s, %(rag_text)s,
            %(links)s, %(attachments)s, %(keywords)s, NOW()
        )
        ON CONFLICT (doc_id) DO UPDATE SET
            title           = EXCLUDED.title,
            type_doc        = EXCLUDED.type_doc,
            category_code   = EXCLUDED.category_code,
            category_name   = EXCLUDED.category_name,
            category_desc   = EXCLUDED.category_desc,
            category_order  = EXCLUDED.category_order,
            source_url      = EXCLUDED.source_url,
            time_create     = EXCLUDED.time_create,
            status          = EXCLUDED.status,
            creator_id      = EXCLUDED.creator_id,
            html            = EXCLUDED.html,
            text            = EXCLUDED.text,
            rag_text        = EXCLUDED.rag_text,
            links           = EXCLUDED.links,
            attachments     = EXCLUDED.attachments,
            keywords        = EXCLUDED.keywords,
            crawled_at      = NOW();
    """
    import json
    params = {
        "doc_id":           str(article.get("doc_id", "")),
        "title":            article.get("title", ""),
        "type_doc":         article.get("type_doc"),
        "category_code":    article.get("category_code"),
        "category_name":    article.get("category_name"),
        "category_desc":    article.get("category_desc"),
        "category_order":   article.get("category_order"),
        "source_url":       article.get("source_url"),
        "time_create":      article.get("time_create"),
        "status":           article.get("status"),
        "creator_id":       str(article.get("creator_id", "")) if article.get("creator_id") else None,
        "html":             article.get("html", ""),
        "text":             article.get("text", ""),
        "rag_text":         article.get("rag_text", ""),
        "links":            psycopg2.extras.Json(article.get("links", [])),
        "attachments":      psycopg2.extras.Json(article.get("attachments", [])),
        "keywords":         psycopg2.extras.Json(article.get("keywords", {})),
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)


def upsert_chunks(conn: psycopg2.extensions.connection, chunks: list[dict]) -> None:
    """
    Bulk insert hoặc update danh sách chunks vào bảng `rag_chunks`.
    Dùng executemany với ON CONFLICT DO UPDATE.
    """
    if not chunks:
        return

    sql = """
        INSERT INTO rag_chunks (
            id, parent_id, doc_id, chunk_index, chunk_type,
            is_attachment, attachment_name, text, content_hash, metadata, created_at
        ) VALUES (
            %(id)s, %(parent_id)s, %(doc_id)s, %(chunk_index)s, %(chunk_type)s,
            %(is_attachment)s, %(attachment_name)s, %(text)s, %(content_hash)s, %(metadata)s, NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            parent_id       = EXCLUDED.parent_id,
            doc_id          = EXCLUDED.doc_id,
            chunk_index     = EXCLUDED.chunk_index,
            chunk_type      = EXCLUDED.chunk_type,
            is_attachment   = EXCLUDED.is_attachment,
            attachment_name = EXCLUDED.attachment_name,
            text            = EXCLUDED.text,
            content_hash    = EXCLUDED.content_hash,
            metadata        = EXCLUDED.metadata,
            embedded_at     = CASE
                                WHEN rag_chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash THEN NULL
                                ELSE rag_chunks.embedded_at
                              END,
            created_at      = NOW();
    """
    params_list = []
    for chunk in chunks:
        # parent_id có dạng "hust_sotay_{doc_id}", doc_id thuần không có prefix
        parent_id = chunk.get("parent_id", "")
        doc_id_raw = parent_id.replace("hust_sotay_", "", 1) if parent_id.startswith("hust_sotay_") else parent_id

        params_list.append({
            "id":               chunk["id"],
            "parent_id":        parent_id,
            "doc_id":           doc_id_raw,
            "chunk_index":      chunk.get("chunk_index", 0),
            "chunk_type":       chunk.get("chunk_type", "body"),
            "is_attachment":    chunk.get("is_attachment", False),
            "attachment_name":  chunk.get("attachment_name"),
            "text":             chunk.get("text", ""),
            "content_hash":     chunk.get("metadata", {}).get("content_hash"),
            "metadata":         psycopg2.extras.Json(chunk.get("metadata", {})),
        })

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, params_list, page_size=100)


''' Embedding helpers '''
def get_unembedded_chunks(
    conn: psycopg2.extensions.connection,
    limit: int | None = None,
    category_code: str | None = None,
) -> list[dict]:
    """
    Trả về danh sách chunks chưa có embedding (embedding IS NULL).
    
    Args:
        conn: PostgreSQL connection.
        limit: Số lượng chunks tối đa cần lấy (None = tất cả).
        category_code: Lọc theo mã chuyên mục (HT, HC, HB, DS, KN, HD).
    
    Returns:
        List[dict] với các key: id, text, metadata.
    """
    conditions = ["embedded_at IS NULL"]
    params: dict = {}

    if category_code:
        conditions.append("metadata->>'category_code' = %(category_code)s")
        params["category_code"] = category_code

    where_clause = " AND ".join(conditions)
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT id, text, metadata
        FROM rag_chunks
        WHERE {where_clause}
        ORDER BY created_at ASC
        {limit_clause};
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def mark_chunks_as_embedded(
    conn: psycopg2.extensions.connection,
    chunk_ids: list[str],
) -> int:
    """
    Đánh dấu các chunks đã được upsert vào Qdrant bằng cách cập nhật `embedded_at`.
    Vector không lưu trong PostgreSQL – lưu trong Qdrant.

    Args:
        conn: PostgreSQL connection.
        chunk_ids: Danh sách chunk ID đã upsert thành công vào Qdrant.

    Returns:
        Số rows đã được update.
    """
    if not chunk_ids:
        return 0

    sql = """
        UPDATE rag_chunks
        SET embedded_at = NOW()
        WHERE id = %(id)s;
    """
    params_list = [{"id": cid} for cid in chunk_ids]

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, params_list, page_size=200)

    return len(chunk_ids)


# Giữ tên cũ như một alias để backward-compatible
upsert_embeddings = mark_chunks_as_embedded


def get_embedding_stats(conn: psycopg2.extensions.connection) -> dict:
    """
    Trả về thống kê trạng thái embedding của bảng rag_chunks.
    Tracking dựa trên `embedded_at` (Qdrant Edition – không có cột vector).

    Returns:
        dict với các key: total, embedded, unembedded, pct_done.
    """
    sql = """
        SELECT
            COUNT(*)                                        AS total,
            COUNT(*) FILTER (WHERE embedded_at IS NOT NULL) AS embedded,
            COUNT(*) FILTER (WHERE embedded_at IS NULL)     AS unembedded
        FROM rag_chunks;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        row = dict(cur.fetchone())

    total = row["total"] or 0
    embedded = row["embedded"] or 0
    row["pct_done"] = round(embedded / total * 100, 1) if total else 0.0
    return row


def create_pipeline_run(conn: psycopg2.extensions.connection, **kwargs) -> int:
    """Tạo một bản ghi pipeline_run mới và trả về ID của nó."""
    sql = """
        INSERT INTO pipeline_runs (
            started_at, fetch_attachments, use_detail_endpoint,
            enrich_local, status
        ) VALUES (NOW(), %(fetch_attachments)s, %(use_detail_endpoint)s,
                  %(enrich_local)s, 'running')
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "fetch_attachments":    kwargs.get("fetch_attachments", False),
            "use_detail_endpoint":  kwargs.get("use_detail_endpoint", True),
            "enrich_local":         kwargs.get("enrich_local", False),
        })
        conn.commit()
        return cur.fetchone()[0]


def finish_pipeline_run(
    conn: psycopg2.extensions.connection,
    run_id: int,
    *,
    total_articles: int,
    total_chunks: int,
    total_api_items: int,
    attachment_stats: dict,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    """Cập nhật pipeline_run sau khi hoàn thành hoặc gặp lỗi."""
    sql = """
        UPDATE pipeline_runs SET
            finished_at      = NOW(),
            total_articles   = %(total_articles)s,
            total_chunks     = %(total_chunks)s,
            total_api_items  = %(total_api_items)s,
            attachment_stats = %(attachment_stats)s,
            status           = %(status)s,
            error_message    = %(error_message)s
        WHERE id = %(run_id)s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "run_id":            run_id,
            "total_articles":    total_articles,
            "total_chunks":      total_chunks,
            "total_api_items":   total_api_items,
            "attachment_stats":  psycopg2.extras.Json(attachment_stats),
            "status":            status,
            "error_message":     error_message,
        })
        conn.commit()
