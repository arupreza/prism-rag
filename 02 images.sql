-- PRISM-RAG migration: image/figure support
-- Run ONCE, OUTSIDE a transaction block. Postgres forbids using a freshly
-- added enum value in the same transaction that adds it, so keep these as
-- standalone autocommit statements (psql runs them this way by default).

ALTER TYPE content_type_t ADD VALUE IF NOT EXISTS 'image';

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_path TEXT;

-- Optional: quickly find figure rows for display / audits.
CREATE INDEX IF NOT EXISTS idx_chunks_image ON chunks(image_path)
    WHERE image_path IS NOT NULL;