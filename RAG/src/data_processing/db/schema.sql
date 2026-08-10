-- HUST RAG Pipeline – PostgreSQL Schema  (Qdrant Edition)
-- PostgreSQL lưu text + metadata + full-text search.
-- Vector embedding được lưu trong Qdrant (local mode).

-- Bảng articles: Lưu bài viết gốc đã làm sạch (1 dòng = 1 bài)
CREATE TABLE IF NOT EXISTS articles (
    doc_id          TEXT PRIMARY KEY,           -- DocumentID từ API HUST
    title           TEXT        NOT NULL,        -- Tiêu đề đã chuẩn hóa
    type_doc        TEXT,                        -- Loại tài liệu (TypeDoc)
    category_code   TEXT,                        -- Mã chuyên mục: HT, HC, HB, DS, KN, HD
    category_name   TEXT,                        -- Tên chuyên mục
    category_desc   TEXT,                        -- Mô tả chuyên mục
    category_order  INT,                         -- Thứ tự chuyên mục
    source_url      TEXT,                        -- URL trang bài viết
    time_create     TEXT,                        -- Thời gian tạo (chuỗi từ API)
    status          INT,                         -- Status từ API
    creator_id      TEXT,                        -- CreaterID từ API
    html            TEXT,                        -- HTML gốc (Description)
    text            TEXT,                        -- Text đã parse từ HTML
    rag_text        TEXT,                        -- Text tổng hợp dùng cho RAG
    links           JSONB       DEFAULT '[]',    -- Danh sách link trong bài
    attachments     JSONB       DEFAULT '[]',    -- Danh sách file đính kèm đã chuẩn hóa
    keywords        JSONB       DEFAULT '{}',    -- Emails và tags trích xuất
    crawled_at      TIMESTAMPTZ DEFAULT NOW()    -- Thời điểm crawl lần cuối
);

-- Index để tìm kiếm nhanh theo category
CREATE INDEX IF NOT EXISTS idx_articles_category_code ON articles(category_code);
CREATE INDEX IF NOT EXISTS idx_articles_type_doc ON articles(type_doc);
CREATE INDEX IF NOT EXISTS idx_articles_crawled_at ON articles(crawled_at DESC);


-- Bảng rag_chunks: Lưu chunks đã cắt đoạn
-- Vector embedding được lưu riêng trong Qdrant (không lưu ở đây)
CREATE TABLE IF NOT EXISTS rag_chunks (
    id              TEXT PRIMARY KEY,            -- chunk ID: hust_sotay_{doc_id}_chunk000
    parent_id       TEXT        NOT NULL,        -- Dạng hust_sotay_{doc_id}
    doc_id          TEXT        REFERENCES articles(doc_id) ON DELETE CASCADE,
    chunk_index     INT         NOT NULL,        -- Thứ tự chunk trong bài
    chunk_type      TEXT        NOT NULL,        -- 'body' hoặc 'attachment'
    is_attachment   BOOLEAN     DEFAULT FALSE,   -- True nếu từ file đính kèm
    attachment_name TEXT,                        -- Tên file đính kèm (nếu có)
    text            TEXT        NOT NULL,        -- Nội dung chunk (kèm prefix tiêu đề)
    content_hash    TEXT,                        -- Hash của nội dung chunk (để trigger re-embed)
    metadata        JSONB       DEFAULT '{}',    -- Metadata đầy đủ: source, url, category...
    embedded_at     TIMESTAMPTZ,                 -- Thời điểm đã upsert vào Qdrant (NULL = chưa embed)
    created_at      TIMESTAMPTZ DEFAULT NOW()    -- Thời điểm insert/update
);

-- Index thông thường
CREATE INDEX IF NOT EXISTS idx_rag_chunks_parent_id     ON rag_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc_id        ON rag_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_chunk_type    ON rag_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_is_attachment ON rag_chunks(is_attachment);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_created_at    ON rag_chunks(created_at DESC);

-- Partial index: tìm nhanh chunks chưa được embed vào Qdrant
CREATE INDEX IF NOT EXISTS idx_rag_chunks_not_embedded
    ON rag_chunks(created_at ASC)
    WHERE embedded_at IS NULL;

-- Full-text search index (tiếng Việt dùng 'simple' config)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_rag_chunks_text_trgm
    ON rag_chunks USING GIN (text gin_trgm_ops);

-- GIN index trên metadata JSONB để query nhanh theo category
CREATE INDEX IF NOT EXISTS idx_rag_chunks_metadata ON rag_chunks USING GIN (metadata);


-- Bảng pipeline_runs: Log lịch sử mỗi lần chạy pipeline
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    total_articles  INT         DEFAULT 0,
    total_chunks    INT         DEFAULT 0,
    total_api_items INT         DEFAULT 0,
    fetch_attachments BOOLEAN   DEFAULT FALSE,
    use_detail_endpoint BOOLEAN DEFAULT TRUE,
    enrich_local    BOOLEAN     DEFAULT FALSE,
    status          TEXT        DEFAULT 'running', -- 'running' | 'success' | 'error'
    error_message   TEXT,
    attachment_stats JSONB      DEFAULT '{}',
    summary         JSONB       DEFAULT '{}'
);
