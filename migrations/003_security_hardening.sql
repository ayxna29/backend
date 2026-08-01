-- Security hardening
-- Purpose: (1) close the RLS gap on `users` — it was the only user-owned
-- table without RLS, meaning any authenticated user could read/write any
-- other user's row via the client's publishable key; (2) pin search_path
-- on every function flagged by Supabase's database linter
-- (function_search_path_mutable), which otherwise leaves them open to
-- search_path hijacking if an attacker can create objects earlier in the
-- resolution path than expected.

-- ============================================
-- RLS: users
-- ============================================
-- The Flutter client reads/writes its own row directly (signup.dart,
-- optimization.dart) using the publishable key + the user's session, the
-- same pattern already used (and already RLS-protected) for user_tags,
-- tag_contexts, and user_favorites. The backend's service-role client is
-- unaffected by RLS either way.

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS users_policy ON users;
CREATE POLICY users_policy ON users
  FOR ALL
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

GRANT SELECT, INSERT, UPDATE ON users TO authenticated;

-- ============================================
-- Pin search_path on every flagged function
-- ============================================
-- Looked up dynamically via pg_proc rather than hardcoding each function's
-- argument signature, so this only ever touches function CONFIGURATION
-- (never the body/logic) and stays correct even if a signature changes.
--
-- Pinned to `public, pg_temp` rather than an empty string: `current_auth_uid`
-- and `match_user_flashcards` aren't defined anywhere in this repo (created
-- directly in the Supabase SQL editor), so their bodies are unknown here —
-- if either references a public-schema table/type unqualified (e.g.
-- `flashcards` instead of `public.flashcards`), an empty search_path would
-- break them. `public, pg_temp` still satisfies the linter (a FIXED,
-- non-mutable path instead of an inherited/mutable one) while keeping
-- unqualified public-schema references working. Test both functions after
-- running this — if either still breaks, its body has a non-public schema
-- reference that needs to be added to this path explicitly.

DO $$
DECLARE
  func RECORD;
BEGIN
  FOR func IN
    SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS args
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public'
      AND p.proname IN ('update_updated_at_column', 'current_auth_uid', 'match_user_flashcards')
  LOOP
    EXECUTE format('ALTER FUNCTION public.%I(%s) SET search_path = ''public, pg_temp''', func.proname, func.args);
  END LOOP;
END $$;
