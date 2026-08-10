-- ─────────────────────────────────────────────────────────────
--  Academic Recommendation System - Database Schema
-- ─────────────────────────────────────────────────────────────

-- Users
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        VARCHAR(100) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(255),
    department      VARCHAR(150),        -- e.g. "Computer Science", "Biology"
    role            VARCHAR(50),         -- e.g. "professor", "student", "researcher"
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Interest tags master table
-- category = broad domain (e.g. "AI", "Biology", "Physics")
-- name     = specific interest (e.g. "Machine Learning", "CRISPR", "Quantum Computing")
CREATE TABLE IF NOT EXISTS tags (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) UNIQUE NOT NULL,
    category    VARCHAR(100) NOT NULL
);

-- User selected interest tags (max 5)
CREATE TABLE IF NOT EXISTS user_tags (
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    tag_id      INT  REFERENCES tags(id)  ON DELETE CASCADE,
    PRIMARY KEY (user_id, tag_id)
);

-- Papers published by users
CREATE TABLE IF NOT EXISTS papers (
    id              SERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    title           VARCHAR(500) NOT NULL,
    abstract        TEXT,
    published_at    DATE
);

-- Keywords extracted from each paper (from keywords/tags field)
CREATE TABLE IF NOT EXISTS paper_keywords (
    paper_id    INT REFERENCES papers(id) ON DELETE CASCADE,
    keyword     VARCHAR(100) NOT NULL,
    PRIMARY KEY (paper_id, keyword)
);

-- Activity tracking
CREATE TABLE IF NOT EXISTS user_activity (
    user_id         UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    last_active_at  TIMESTAMP DEFAULT NOW(),
    activity_score  FLOAT DEFAULT 0.5
);

-- ─── Indexes ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_user_tags_user_id       ON user_tags(user_id);
CREATE INDEX IF NOT EXISTS idx_user_tags_tag_id        ON user_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_category           ON tags(category);
CREATE INDEX IF NOT EXISTS idx_papers_user_id          ON papers(user_id);
CREATE INDEX IF NOT EXISTS idx_paper_keywords_paper_id ON paper_keywords(paper_id);
CREATE INDEX IF NOT EXISTS idx_users_department        ON users(department);

-- ─── Auto-create activity row on new user ─────────────────────
CREATE OR REPLACE FUNCTION create_user_activity()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_activity (user_id, last_active_at, activity_score)
    VALUES (NEW.id, NOW(), 0.5)
    ON CONFLICT DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_create_user_activity ON users;
CREATE TRIGGER trg_create_user_activity
    AFTER INSERT ON users
    FOR EACH ROW EXECUTE FUNCTION create_user_activity();