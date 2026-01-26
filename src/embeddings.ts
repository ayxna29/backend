import OpenAI from 'openai';
import { getSupabase } from './supabase.js';

let embeddingClient: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (embeddingClient) return embeddingClient;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Missing OPENAI_API_KEY');
  embeddingClient = new OpenAI({ apiKey: key });
  return embeddingClient;
}

// Model & dimension (text-embedding-3-small => 1536)
export const EMBEDDING_MODEL = 'text-embedding-3-small';
export const EMBEDDING_DIM = 1536;
export const DUPLICATE_SIM_THRESHOLD = 0.90; // cosine similarity

const FETCH_TIMEOUT_MS = Number(process.env.SUPABASE_FETCH_TIMEOUT_MS || 3000);
const EMB_CACHE_TTL_MS = Number(process.env.EMBEDDING_CACHE_TTL_MS || 300000); // 5 min
const MAX_EXISTING_ROWS = Number(process.env.EMBEDDING_EXISTING_LIMIT || 400);
interface CachedEmbeds {
  expires: number;
  vectors: number[][];
}
const embedCache = new Map<string, CachedEmbeds>();

export async function embedBatch(texts: string[]): Promise<number[][]> {
  if (!texts.length) return [];
  const openai = getOpenAI();
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts
  });
  return res.data.map(v => v.embedding as number[]);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

export async function fetchExistingEmbeddings(userId: string): Promise<number[][]> {
  // Cached?
  const cached = embedCache.get(userId);
  const now = Date.now();
  if (cached && cached.expires > now) {
    return cached.vectors;
  }

  const supabase = getSupabase();
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    // Limit rows to cap latency
    const { data, error } = await supabase
      .from('flashcards')
      .select('embedding')
      .eq('user_id', userId)
      .not('embedding', 'is', null)
      .order('created_at', { ascending: false })
      .limit(MAX_EXISTING_ROWS);
    clearTimeout(t);
    if (error) {
      console.warn('[embeddings fetch error]', error.message);
      return [];
    }
    const vectors = (data || [])
      .map(r => r.embedding)
      .filter(v => Array.isArray(v) && v.length > 0);
    embedCache.set(userId, { expires: now + EMB_CACHE_TTL_MS, vectors });
    return vectors;
  } catch (e: any) {
    clearTimeout(t);
    console.warn('[embeddings fetch failed]', e?.message || e);
    return [];
  }
}

export function filterDuplicates(
  newEmbeds: number[][],
  existing: number[][]
): { keep: boolean[]; removed: number } {
  if (!existing.length) return { keep: newEmbeds.map(() => true), removed: 0 };
  const keep: boolean[] = [];
  let removed = 0;
  for (const e of newEmbeds) {
    let maxSim = 0;
    for (const ex of existing) {
      const s = cosine(e, ex);
      if (s > maxSim) maxSim = s;
      if (maxSim >= DUPLICATE_SIM_THRESHOLD) break;
    }
    if (maxSim >= DUPLICATE_SIM_THRESHOLD) {
      keep.push(false);
      removed++;
    } else {
      keep.push(true);
    }
  }
  return { keep, removed };
}