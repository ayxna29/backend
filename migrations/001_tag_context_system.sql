-- ============================================
-- Tag Context System Database Migration (Safe Version)
-- Created: 2025-10-24
-- Purpose: Enable tag-based context management for personalized flashcard generation
-- ============================================

-- Drop existing tables safely (constraints + triggers auto-removed)
DROP TABLE IF EXISTS user_context_sentences CASCADE;
DROP TABLE IF EXISTS user_context_tags CASCADE;

-- ============================================
-- Table: user_context_tags
-- ============================================
CREATE TABLE user_context_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tag_name TEXT NOT NULL,
  emoji TEXT,
  color TEXT,
  is_active BOOLEAN DEFAULT true,
  display_order INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT user_context_tags_unique_tag UNIQUE(user_id, tag_name),
  CONSTRAINT user_context_tags_tag_name_len CHECK (char_length(tag_name) BETWEEN 1 AND 50),
  CONSTRAINT user_context_tags_emoji_len CHECK (emoji IS NULL OR char_length(emoji) <= 10)
);

CREATE INDEX idx_user_context_tags_user_id ON user_context_tags(user_id);
CREATE INDEX idx_user_context_tags_order ON user_context_tags(user_id, display_order);
CREATE INDEX idx_user_context_tags_active ON user_context_tags(user_id, is_active) WHERE is_active = true;

-- ============================================
-- Table: user_context_sentences
-- ============================================
CREATE TABLE user_context_sentences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tag_id UUID NOT NULL REFERENCES user_context_tags(id) ON DELETE CASCADE,
  sentence_text TEXT NOT NULL,
  display_order INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT user_context_sentences_text_len CHECK (char_length(sentence_text) BETWEEN 1 AND 500)
);

CREATE INDEX idx_user_context_sentences_tag_id ON user_context_sentences(tag_id);
CREATE INDEX idx_user_context_sentences_order ON user_context_sentences(tag_id, display_order);

-- ============================================
-- Function: Update updated_at timestamp
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop & recreate triggers with unique names
DROP TRIGGER IF EXISTS trg_update_user_context_tags ON user_context_tags;
CREATE TRIGGER trg_update_user_context_tags
BEFORE UPDATE ON user_context_tags
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_update_user_context_sentences ON user_context_sentences;
CREATE TRIGGER trg_update_user_context_sentences
BEFORE UPDATE ON user_context_sentences
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Grant Permissions
-- ============================================
GRANT ALL ON user_context_tags TO service_role;
GRANT ALL ON user_context_sentences TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_context_tags TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_context_sentences TO authenticated;

-- ============================================
-- Row Level Security (RLS) Policies
-- ============================================
ALTER TABLE user_context_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_context_sentences ENABLE ROW LEVEL SECURITY;

-- user_context_tags RLS
DROP POLICY IF EXISTS user_context_tags_policy_all ON user_context_tags;
CREATE POLICY user_context_tags_policy_all ON user_context_tags
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- user_context_sentences RLS
DROP POLICY IF EXISTS user_context_sentences_policy_all ON user_context_sentences;
CREATE POLICY user_context_sentences_policy_all ON user_context_sentences
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM user_context_tags
      WHERE user_context_tags.id = user_context_sentences.tag_id
      AND user_context_tags.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM user_context_tags
      WHERE user_context_tags.id = user_context_sentences.tag_id
      AND user_context_tags.user_id = auth.uid()
    )
  );
