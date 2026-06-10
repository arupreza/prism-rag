-- PRISM-RAG v2 schema: 3-domain RAPTOR tree on pgvector
-- Domains: immigration, trading, ai
-- Parent/child chunking: children are searchable, parents are storage-only.
--
-- IDEMPOTENT + NON-DESTRUCTIVE: safe to re-run on a populated database. It only
-- creates what is missing and evolves the schema in place; it NEVER drops data.
-- (This is why there is no separate migrations/ file — everything lives here.)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Enums ────────────────────────────────────────────────────────────────────
-- CREATE TYPE has no IF NOT EXISTS, so guard with a DO block.
DO $$ BEGIN
    CREATE TYPE domain_t AS ENUM ('immigration', 'trading', 'ai');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE content_type_t AS ENUM ('text', 'code', 'image', 'mixed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Evolve a content_type_t that predates 'image' (no-op on a fresh enum).
ALTER TYPE content_type_t ADD VALUE IF NOT EXISTS 'image';

-- ── Source documents (PDFs / source files) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    domain      domain_t      NOT NULL,
    source_path TEXT          NOT NULL UNIQUE,
    title       TEXT,
    n_pages     INT,
    sha256      CHAR(64)      NOT NULL,
    metadata    JSONB         DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ   DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);

-- ── Unified chunk table ──────────────────────────────────────────────────────
--   level = 0, parent_chunk_id IS NULL,     is_searchable = FALSE -> PARENT (paragraph)
--   level = 0, parent_chunk_id IS NOT NULL, is_searchable = TRUE  -> CHILD shard
--   level >= 1,                             is_searchable = TRUE  -> RAPTOR summary node
--   content_type = 'image' -> caption/explanation row, image_path -> figure on disk
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    domain          domain_t       NOT NULL,
    level           INT            NOT NULL,
    cluster_id      INT,
    parent_chunk_id BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
    is_searchable   BOOLEAN        NOT NULL DEFAULT TRUE,
    parent_ids      BIGINT[]       DEFAULT '{}',   -- RAPTOR multi-membership (soft GMM)
    children_ids    BIGINT[]       DEFAULT '{}',   -- summary -> its child summary/leaf ids
    content         TEXT           NOT NULL,
    content_type    content_type_t NOT NULL DEFAULT 'text',
    language        TEXT,
    page_start      INT,
    page_end        INT,
    token_count     INT,
    image_path      TEXT,                          -- filesystem path to figure (content_type='image')
    embedding       vector(1024)   NOT NULL,
    metadata        JSONB          DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ    DEFAULT now()
);

-- Columns that may be missing on a chunks table created before this revision.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_path TEXT;

-- ── Policy CHECKs (ADD CONSTRAINT has no IF NOT EXISTS -> guard) ─────────────
DO $$ BEGIN
    ALTER TABLE chunks ADD CONSTRAINT chk_trading_no_code
        CHECK (NOT (domain = 'trading' AND content_type = 'code'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE chunks ADD CONSTRAINT chk_language_only_for_code
        CHECK ((content_type = 'code') OR (language IS NULL));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ── Standard indexes ─────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chunks_domain_level ON chunks(domain, level);
CREATE INDEX IF NOT EXISTS idx_chunks_document     ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_parent       ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_image        ON chunks(image_path) WHERE image_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_trgm         ON chunks USING gin (content gin_trgm_ops);

-- ── HNSW per domain, restricted to searchable rows (children + summaries) ────
CREATE INDEX IF NOT EXISTS idx_chunks_emb_imm ON chunks USING hnsw (embedding vector_cosine_ops)
    WHERE domain = 'immigration' AND is_searchable;
CREATE INDEX IF NOT EXISTS idx_chunks_emb_trd ON chunks USING hnsw (embedding vector_cosine_ops)
    WHERE domain = 'trading'     AND is_searchable;
CREATE INDEX IF NOT EXISTS idx_chunks_emb_ai  ON chunks USING hnsw (embedding vector_cosine_ops)
    WHERE domain = 'ai'          AND is_searchable;