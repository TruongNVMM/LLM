-- ===========================================================================
-- HUST RAG Pipeline – PostgreSQL Schema
-- ===========================================================================
-- Tạo tất cả bảng nếu chưa tồn tại (idempotent).
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Bảng articles: Lưu bài viết gốc đã làm sạch (1 dòng = 1 bài)
-- ---------------------------------------------------------------------------
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
    keywords        JSONB       DEFAULT '{}',    -- Emails và tags trích xuất
    crawled_at      TIMESTAMPTZ DEFAULT NOW()    -- Thời điểm crawl lần cuối
);

-- Index để tìm kiếm nhanh theo category
CREATE INDEX IF NOT EXISTS idx_articles_category_code ON articles(category_code);
CREATE INDEX IF NOT EXISTS idx_articles_type_doc ON articles(type_doc);
CREATE INDEX IF NOT EXISTS idx_articles_crawled_at ON articles(crawled_at DESC);

-- ---------------------------------------------------------------------------
-- Bảng rag_chunks: Lưu chunks đã cắt đoạn (sẵn sàng cho vector embedding)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag_chunks (
    id              TEXT PRIMARY KEY,            -- chunk ID: hust_sotay_{doc_id}_chunk000
    parent_id       TEXT        NOT NULL,        -- Dạng hust_sotay_{doc_id}
    doc_id          TEXT        REFERENCES articles(doc_id) ON DELETE CASCADE, -- Khóa ngoại trỏ đến articles(doc_id)
    chunk_index     INT         NOT NULL,        -- Thứ tự chunk trong bài
    chunk_type      TEXT        NOT NULL,        -- 'body' hoặc 'attachment'
    is_attachment   BOOLEAN     DEFAULT FALSE,   -- True nếu từ file đính kèm
    attachment_name TEXT,                        -- Tên file đính kèm (nếu có)
    text            TEXT        NOT NULL,        -- Nội dung chunk (kèm prefix tiêu đề)
    metadata        JSONB       DEFAULT '{}',    -- Metadata đầy đủ: source, url, category, links...
    created_at      TIMESTAMPTZ DEFAULT NOW()    -- Thời điểm insert/update
);

-- Index để tìm kiếm nhanh
CREATE INDEX IF NOT EXISTS idx_rag_chunks_parent_id    ON rag_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc_id       ON rag_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_chunk_type   ON rag_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_is_attachment ON rag_chunks(is_attachment);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_created_at   ON rag_chunks(created_at DESC);

-- Full-text search index trên nội dung chunk (tiếng Việt dùng 'simple' hoặc 'pg_trgm')
CREATE INDEX IF NOT EXISTS idx_rag_chunks_text_trgm
    ON rag_chunks USING GIN (text gin_trgm_ops);

-- GIN index trên metadata JSONB để query nhanh
CREATE INDEX IF NOT EXISTS idx_rag_chunks_metadata ON rag_chunks USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- Bảng pipeline_runs: Log lịch sử mỗi lần chạy pipeline
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ DEFAULT NOW(),   -- Thời điểm bắt đầu
    finished_at     TIMESTAMPTZ,                 -- Thời điểm kết thúc
    total_articles  INT         DEFAULT 0,       -- Số bài viết đã xử lý
    total_chunks    INT         DEFAULT 0,       -- Số chunks đã lưu
    total_api_items INT         DEFAULT 0,       -- Tổng items từ API (trước khi lọc)
    fetch_attachments BOOLEAN   DEFAULT FALSE,   -- Có fetch file đính kèm không
    use_detail_endpoint BOOLEAN DEFAULT TRUE,    -- Có gọi GetWebTitleInfo không
    enrich_local    BOOLEAN     DEFAULT FALSE,   -- Có enrich từ data_fetch/ không
    status          TEXT        DEFAULT 'running', -- 'running' | 'success' | 'error'
    error_message   TEXT,                        -- Thông báo lỗi (nếu có)
    attachment_stats JSONB      DEFAULT '{}',    -- Thống kê fetch attachment
    summary         JSONB       DEFAULT '{}'     -- Thống kê tổng hợp
);
