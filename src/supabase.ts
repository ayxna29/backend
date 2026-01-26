import { createClient, SupabaseClient } from '@supabase/supabase-js';

let adminClient: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!adminClient) {
    const url = process.env.SUPABASE_URL!;
    // Prefer the SERVICE_ROLE key but fall back to SERVICE_KEY for backwards-compatibility
    const service = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!;
    if (!service) throw new Error('Missing Supabase service key: set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY');
    adminClient = createClient(url, service, { auth: { autoRefreshToken: false, persistSession: false } });
  }
  return adminClient;
}

export async function verifyJwt(token?: string) {
  if (!token) return null;
  try {
    const { data, error } = await getSupabase().auth.getUser(token);
    if (error || !data?.user) return null;
    return { id: data.user.id };
  } catch {
    return null;
  }
}