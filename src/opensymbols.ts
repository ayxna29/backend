// openSymbols.ts
//
// Thin client for the OpenSymbols API (https://www.opensymbols.org/docs).
// Used to resolve a real symbol image for a generated flashcard's answer,
// exposed to callers as an `assetUrl`. This is a best-effort enhancement,
// never a hard dependency — every function here fails soft (returns null)
// on any error, missing config, or timeout. index.ts's local Mulberry
// symbol matching (`asset_filename`) already covers the fallback case.

const OPENSYMBOLS_BASE_URL = process.env.OPENSYMBOLS_BASE_URL || 'https://www.opensymbols.org';

// We don't know the exact env var name already sitting in your .env, so we
// check the common variants. Whichever is set first wins. If none are set,
// OpenSymbols is simply disabled (isOpenSymbolsConfigured() -> false) and
// everything falls back to the local symbol matching, same as before.
const OPENSYMBOLS_SECRET =
  process.env.OPENSYMBOLS_SECRET ||
  process.env.OPEN_SYMBOLS_SECRET ||
  process.env.OPENSYMBOLS_API_SECRET ||
  process.env.OPENSYMBOLS_SHARED_SECRET ||
  '';

if (OPENSYMBOLS_SECRET) {
  console.log('[OPENSYMBOLS] secret found, symbol lookups enabled');
} else {
  console.warn('[OPENSYMBOLS] no secret found in env (checked OPENSYMBOLS_SECRET, OPEN_SYMBOLS_SECRET, OPENSYMBOLS_API_SECRET, OPENSYMBOLS_SHARED_SECRET) — symbol lookups disabled, local asset matching only');
}

// The docs describe tokens as "short-lived" without publishing an exact
// TTL. We proactively refresh well ahead of any likely expiry, and also
// react immediately to a 401 `token_expired` response regardless of this
// timer, so an overly-optimistic TTL here just costs one extra 401+retry.
const TOKEN_TTL_MS = Number(process.env.OPENSYMBOLS_TOKEN_TTL_MS || 10 * 60 * 1000); // 10 min

let cachedToken: string | null = null;
let cachedTokenExpiresAt = 0;
let tokenFetchInFlight: Promise<string | null> | null = null;

// Query -> resolved image URL (or null for "looked up, nothing found").
// AAC vocabulary is small and repetitive (i, am, good, help, more...) so
// this cache does a lot of work keeping repeat lookups instant and off
// the network entirely.
const symbolCache = new Map<string, string | null>();
const MAX_CACHE_ENTRIES = 2000;

function cacheSet(key: string, value: string | null) {
  if (symbolCache.size >= MAX_CACHE_ENTRIES) {
    const firstKey = symbolCache.keys().next().value;
    if (firstKey !== undefined) symbolCache.delete(firstKey);
  }
  symbolCache.set(key, value);
}

export function isOpenSymbolsConfigured(): boolean {
  return !!OPENSYMBOLS_SECRET;
}

async function fetchNewToken(): Promise<string | null> {
  if (!OPENSYMBOLS_SECRET) return null;
  try {
    const url = `${OPENSYMBOLS_BASE_URL}/api/v2/token?secret=${encodeURIComponent(OPENSYMBOLS_SECRET)}`;
    const resp = await fetch(url, { method: 'POST' });
    if (!resp.ok) {
      console.warn('[OPENSYMBOLS] token request failed', { status: resp.status });
      return null;
    }
    const data: any = await resp.json().catch(() => null);
    const token = data?.access_token;
    if (!token) {
      console.warn('[OPENSYMBOLS] token response missing access_token', data);
      return null;
    }
    cachedToken = token;
    cachedTokenExpiresAt = Date.now() + TOKEN_TTL_MS;
    return token;
  } catch (e) {
    console.warn('[OPENSYMBOLS] token fetch error', e);
    return null;
  }
}

async function getAccessToken(forceRefresh = false): Promise<string | null> {
  if (!OPENSYMBOLS_SECRET) return null;
  if (!forceRefresh && cachedToken && Date.now() < cachedTokenExpiresAt) {
    return cachedToken;
  }
  // Coalesce concurrent refreshes (e.g. several cards resolving at once
  // right as the token expires) into a single in-flight request instead
  // of hammering the token endpoint.
  if (!tokenFetchInFlight) {
    tokenFetchInFlight = fetchNewToken().finally(() => { tokenFetchInFlight = null; });
  }
  return tokenFetchInFlight;
}

interface OpenSymbolsResult {
  image_url?: string;
  [key: string]: any;
}

async function searchOnce(
  query: string,
  locale: string,
  token: string
): Promise<{ status: number; results: OpenSymbolsResult[] | null; tokenExpired: boolean; throttled: boolean }> {
  const url = `${OPENSYMBOLS_BASE_URL}/api/v2/symbols?q=${encodeURIComponent(query)}&locale=${encodeURIComponent(locale)}&access_token=${encodeURIComponent(token)}`;
  const resp = await fetch(url);
  if (resp.status === 401) return { status: 401, results: null, tokenExpired: true, throttled: false };
  if (resp.status === 429) return { status: 429, results: null, tokenExpired: false, throttled: true };
  if (!resp.ok) return { status: resp.status, results: null, tokenExpired: false, throttled: false };
  const data = await resp.json().catch(() => null);
  return { status: 200, results: Array.isArray(data) ? data : null, tokenExpired: false, throttled: false };
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), ms);
    p.then((v) => { clearTimeout(timer); resolve(v); })
     .catch(() => { clearTimeout(timer); resolve(null); });
  });
}

/**
 * Look up a symbol image URL for `answer` via the OpenSymbols API.
 * Never throws — returns null if OpenSymbols isn't configured, the lookup
 * times out, is throttled, or no result is found. Callers should always
 * have a local fallback ready regardless of this result.
 */
export async function lookupSymbolUrl(
  answer: string,
  opts: { locale?: string; timeoutMs?: number } = {}
): Promise<string | null> {
  if (!OPENSYMBOLS_SECRET) return null;
  const query = answer.trim().toLowerCase();
  if (!query) return null;

  if (symbolCache.has(query)) return symbolCache.get(query) ?? null;

  const locale = opts.locale || 'en';
  // Lowered from 1200ms: lookups now fire concurrently (see
  // generateFlashcardsStream), so this is the ceiling on the SLOWEST
  // lookup in a batch, not a per-card cost that stacks up sequentially.
  const timeoutMs = opts.timeoutMs ?? 700;

  const run = async (): Promise<string | null> => {
    let token = await getAccessToken();
    if (!token) return null;

    let attempt = await searchOnce(query, locale, token);

    if (attempt.tokenExpired) {
      token = await getAccessToken(true);
      if (!token) return null;
      attempt = await searchOnce(query, locale, token);
    }

    if (attempt.throttled) {
      console.warn('[OPENSYMBOLS] throttled', { query });
      return null;
    }

    if (!attempt.results || !attempt.results.length) return null;
    const best = attempt.results.find(r => !!r.image_url) || null;
    return best?.image_url || null;
  };

  const result = await withTimeout(run(), timeoutMs);
  cacheSet(query, result);
  return result;
}