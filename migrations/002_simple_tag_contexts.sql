-- Simple Tag Context System for Optimization Screen
-- Purpose: Store user-created tags and context sentences for AI generation

-- ============================================
-- Table: user_tags
-- Stores user's custom tags
-- ============================================

CREATE TABLE IF NOT EXISTS user_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tag_name TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  CONSTRAINT unique_user_tag UNIQUE(user_id, tag_name),
  CONSTRAINT tag_name_length CHECK (char_length(tag_name) BETWEEN 1 AND 50)
);

CREATE INDEX idx_user_tags_user_id ON user_tags(user_id);

-- ============================================
-- Table: tag_contexts
-- Stores context sentences under each tag
-- ============================================

CREATE TABLE IF NOT EXISTS tag_contexts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tag_id UUID NOT NULL REFERENCES user_tags(id) ON DELETE CASCADE,
  context_text TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  CONSTRAINT context_length CHECK (char_length(context_text) BETWEEN 1 AND 500)
);

CREATE INDEX idx_tag_contexts_tag_id ON tag_contexts(tag_id);

-- ============================================
-- Row Level Security
-- ============================================

ALTER TABLE user_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE tag_contexts ENABLE ROW LEVEL SECURITY;

-- Users can only access their own tags
CREATE POLICY user_tags_policy ON user_tags
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Users can only access contexts from their own tags
CREATE POLICY tag_contexts_policy ON tag_contexts
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM user_tags
      WHERE user_tags.id = tag_contexts.tag_id
      AND user_tags.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM user_tags
      WHERE user_tags.id = tag_contexts.tag_id
      AND user_tags.user_id = auth.uid()
    )
  );

-- ============================================
-- Migrate existing favorites to user_favorites
-- ============================================

-- Create user_favorites table if it doesn't exist
CREATE TABLE IF NOT EXISTS user_favorites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  word TEXT NOT NULL,
  asset_filename TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  CONSTRAINT unique_user_favorite UNIQUE(user_id, word)
);

CREATE INDEX idx_user_favorites_user_id ON user_favorites(user_id);

ALTER TABLE user_favorites ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_favorites_policy ON user_favorites
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Grant permissions
GRANT ALL ON user_tags TO service_role, authenticated;
GRANT ALL ON tag_contexts TO service_role, authenticated;
GRANT ALL ON user_favorites TO service_role, authenticated;
