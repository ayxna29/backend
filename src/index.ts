import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';

import dotenv from 'dotenv';
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

import express, { Request, Response } from 'express';
import cors from 'cors';

import { randomUUID } from 'node:crypto';
import rateLimit from 'express-rate-limit';
import { z } from 'zod';

import { verifyJwt, getSupabase } from './supabase.js';
import { generateFlashcards } from './generateFlashcards.js';
import { logGenerationComplete } from './logging.js';
import { parseGenerateBody, normalizeTag, MAX_CONTEXT_CHARS, MAX_COUNT } from './validation.js';
import {
  embedBatch,
  fetchExistingEmbeddings,
  filterDuplicates,
  EMBEDDING_MODEL
} from './embeddings.js';
import { sha256Base64 } from './hash.js';
import { inferTagsFromCards } from './tagging.js';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

console.log('[STARTUP] launching flashcard server with PID', process.pid);
console.log('[STARTUP] index.ts loaded');

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_NAME = process.env.MODEL_NAME || 'gpt-5-nano'; // override default model
const PORT = Number(process.env.PORT || 5000);
const DEFAULT_COUNT = 30;
const ENABLE_EMBED = (process.env.ENABLE_EMBEDDING || 'true').toLowerCase() === 'true';

const ANSWER_LENGTHS = new Set(['short', 'medium', 'long', 'mixed']);

console.log('[DEBUG] env flags:', {
  SUPABASE_URL: !!process.env.SUPABASE_URL,
  SUPABASE_SERVICE_KEY: !!process.env.SUPABASE_SERVICE_KEY,
  OPENAI_API_KEY: !!process.env.OPENAI_API_KEY,
  ENABLE_EMBED
});

interface Metrics {
  totalRequests: number;
  generationRequests: number;
  generationSuccess: number;
  generationError: number;
  totalLatencyMs: number;
  duplicatesRemoved: number;
  invalidCardsRemoved: number;
}
const metrics: Metrics = {
  totalRequests: 0,
  generationRequests: 0,
  generationSuccess: 0,
  generationError: 0,
  totalLatencyMs: 0,
  duplicatesRemoved: 0,
  invalidCardsRemoved: 0
};

interface TraceRequest extends Request { traceId?: string; }

function loadSymbolData() {
  try {
    const csvPath = path.resolve(__dirname, '../assets/symbol-info.csv');
    if (!fs.existsSync(csvPath)) {
      console.error('[SYMBOLS] CSV not found:', csvPath);
      return { symbols: [], vocabulary: [] };
    }
    const csvContent = fs.readFileSync(csvPath, 'utf-8');
    const records = parse(csvContent, { columns: true, skip_empty_lines: true });
    const symbols = Array.from(new Set(
      records
        .map((row: any) => {
          const name = row['symbol-en'] || row.symbol || row.Symbol || row.name || row.filename;
          if (!name || !name.trim()) return null;
          const cleaned = name.toLowerCase().trim();
          return cleaned.endsWith('.svg') ? cleaned : `${cleaned}.svg`;
        })
        .filter(Boolean)
    ));
    const vocabulary = symbols.map(s => s.replace('.svg', ''));
    console.log(`[SYMBOLS] Loaded ${symbols.length} symbols`);
    return { symbols, vocabulary };
  } catch (err) {
    console.error('[SYMBOLS] Failed to load:', err);
    return { symbols: [], vocabulary: [] };
  }
}

const { symbols: AVAILABLE_SYMBOLS, vocabulary: MULBERRY_VOCABULARY } = loadSymbolData();

function buildSymbolTokenMap(availableSymbols: string[]) {
  const map = new Map<string, string[]>();
  for (const s of availableSymbols) {
    const base = s.toLowerCase().replace('.svg', '');
    const parts = base.split('_').map(p => p.trim()).filter(Boolean);
    for (const p of parts) {
      const arr = map.get(p) || [];
      if (!arr.includes(s)) arr.push(s);
      map.set(p, arr);
    }
  }
  return map;
}

const SYMBOL_TOKEN_MAP = buildSymbolTokenMap(AVAILABLE_SYMBOLS);

const app = express();
app.use(cors({ origin: true, credentials: true, allowedHeaders: ['Authorization','Content-Type','Accept','Origin'] }));
app.use(express.json());

// Auth middleware
app.use(async (req: any, _res, next) => {
  try {
    const auth = req.headers?.authorization;
    if (!auth) return next();
    const [scheme, token] = auth.split(' ');
    if (scheme?.toLowerCase() !== 'bearer' || !token) return next();
    const user = await verifyJwt(token).catch(() => null);
    if (user) req.user = user;
  } catch {}
  next();
});

// Request logging
app.use((req: TraceRequest, res, next) => {
  const started = Date.now();
  const origEnd = res.end;
  res.end = function (...args: any[]): typeof res {
    const ms = Date.now() - started;
    try {
      console.log(JSON.stringify({ ts: new Date().toISOString(), traceId: req.traceId, method: req.method, path: req.path, status: res.statusCode, ms }));
    } catch {}
    // @ts-ignore
    return origEnd.apply(this, args);
  };
  next();
});

app.use((req: TraceRequest, _res, next) => {
  metrics.totalRequests++;
  req.traceId = randomUUID();
  next();
});

const genLimiter = rateLimit({ windowMs: 60_000, max: 10, standardHeaders: true, legacyHeaders: false, handler: (_req, res) => res.status(429).json({ error: 'Too many requests' }) });
app.use('/generate_flashcards', genLimiter);

app.get('/health', (_req, res) => res.json({status:'ok'}));
app.head('/health', (_req, res) => res.status(200).end());

// Minimal root endpoint for quick deploy health check
app.get('/', (_req, res) => {
  res.send('Backend is running');
});


app.get('/metrics', (_req, res) => {
  const avgLatency = metrics.generationSuccess ? Math.round(metrics.totalLatencyMs / metrics.generationSuccess) : 0;
  res.json({ totalRequests: metrics.totalRequests, generationRequests: metrics.generationRequests, generationSuccess: metrics.generationSuccess, generationError: metrics.generationError, avgLatencyMs: avgLatency, duplicatesRemoved: metrics.duplicatesRemoved, invalidCardsRemoved: metrics.invalidCardsRemoved, embeddingEnabled: ENABLE_EMBED });
});

app.post('/embed', async (req: TraceRequest, res: Response) => {
  try {
    const BodySchema = z.object({ text: z.string().min(1).max(2000) });
    const parsed = BodySchema.parse(req.body || {});
    if (!ENABLE_EMBED) return res.status(400).json({ error: 'Embeddings disabled' });
    const [vec] = await embedBatch([parsed.text]);
    res.json({ model: EMBEDDING_MODEL, length: vec.length, embedding: vec });
  } catch (e: any) {
    if (e?.name === 'ZodError') return res.status(400).json({ error: 'Invalid body', issues: e.issues });
    console.error('embed error:', e);
    res.status(500).json({ error: e?.message || 'Server error' });
  }
});

// Add after /health
app.get('/me', async (req, res) => {
  const auth = req.headers.authorization;
  if (!auth) return res.status(401).json({ error: 'no auth' });
  const token = auth.split(' ')[1];
  const user = await verifyJwt(token);
  if (!user) return res.status(401).json({ error: 'invalid token' });
  res.json({ user_id: user.id });
});

// REPLACE the whole /generate_flashcards handler with this:
app.post('/generate_flashcards', async (req: TraceRequest, res: Response) => {
  const t0 = Date.now();
  metrics.generationRequests++;
  // === timing instrumentation ===
  const stageTimes: { stage: string; ms: number }[] = [];
  const mark = (s: string) => stageTimes.push({ stage: s, ms: Date.now() - t0 });

  let generationId: string | null = null;
  let modelUsed = MODEL_NAME;
  let reusedGenerationRecord = false; // keep single declaration

  try {
    // ---------- Auth ----------
    const auth = req.headers.authorization;
    if (!auth) return res.status(401).json({ error: 'Missing Authorization header' });
    const token = auth.split(' ')[1];
    const user = await verifyJwt(token);
    if (!user) return res.status(401).json({ error: 'Invalid token' });

    // ---------- Parse / clamp ----------
    const rawInput: any = req.body || {}; // keep original for extended options
    const body = parseGenerateBody(rawInput, DEFAULT_COUNT); // sanitized core (context, count, tag maybe)
    let { context, tag, count } = body;

    const promptVersion = Number(
      req.query.prompt_version ||
      rawInput.prompt_version ||
      (body as any)?.prompt_version ||
      1
    ) === 2 ? 2 : 1;

    const requestedCount = count ?? DEFAULT_COUNT;
    const effectiveCount = Math.min(Math.max(requestedCount, 1), MAX_COUNT);

    if (!context || typeof context !== 'string' || !context.trim()) {
      return res.status(400).json({ error: 'context required' });
    }
    context = context.trim();

    if (context.length > MAX_CONTEXT_CHARS) {
      return res.status(400).json({ error: `context exceeds ${MAX_CONTEXT_CHARS} chars` });
    }

    tag = normalizeTag(tag);

    // Extended options (taken from raw input, NOT sanitized body)
    const answerLengthRaw = (
      (req.query.answer_length as string) ??
      rawInput.answer_length ??
      'mixed'
    ).toString().toLowerCase();
    const answerLength = ANSWER_LENGTHS.has(answerLengthRaw) ? answerLengthRaw : 'mixed';

    // Parse reuse (must respect explicit false boolean or string)
    const reuseRaw =
      (req.query.reuse !== undefined ? req.query.reuse :
       rawInput.reuse !== undefined ? rawInput.reuse :
       (body as any)?.reuse);
    function toBool(val: any, def=true) {
      if (val === undefined || val === null) return def;
      if (val === false) return false;
      if (val === true) return true;
      if (typeof val === 'string') {
        const v = val.toLowerCase().trim();
        if (['false','0','no','off'].includes(v)) return false;
        if (['true','1','yes','on'].includes(v)) return true;
      }
      return def;
    }
    const reuse = toBool(reuseRaw, true);

    let incomingTags: string[] = [];
    if (Array.isArray(rawInput.tags)) incomingTags = rawInput.tags;
    if (typeof req.query.tag === 'string') incomingTags.push(req.query.tag as string);
    const inferTagsFlag = toBool(req.query.infer_tags ?? rawInput.infer_tags, false);

    // Sanitize tag list
    incomingTags = incomingTags
      .map(t => t?.toString().trim().toLowerCase())
      .filter(t => !!t && t.length <= 40 && /^[a-z0-9._-]+$/.test(t));

    generationId = randomUUID();

    const advancedOptionsUsed =
      answerLength !== 'mixed' ||
      incomingTags.length > 0 ||
      inferTagsFlag;

    console.log('[GEN OPTIONS]', {
      gen: generationId,
      requestedCount: effectiveCount,
      answerLength,
      incomingTags,
      inferTagsFlag,
      reuseRequested: reuse,
      advancedOptionsUsed
    });
    mark('parsed');
    // ---------- Reuse logic ----------
    let attemptedReuse = false;
    if (reuse && !advancedOptionsUsed) {
      attemptedReuse = true;
      console.log('[REUSE CHECK] eligible simple path');
      try {
        const contextHash = sha256Base64(context);
        const supabase = getSupabase();
        const sinceIso = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
        
        // ADD THIS BLOCK BEFORE THE SELECT QUERY:
        // Ensure generation row exists with context_text BEFORE checking for reuse
        const { error: ensureErr } = await supabase
          .from('flashcard_generations')
          .upsert(
            [{
              id: generationId,
              user_id: user.id,
              context_hash: contextHash,
              context_text: context,  // FIX: add this
              prompt_version: promptVersion,
              created_at: new Date().toISOString(),
            }],
            { onConflict: 'user_id,context_hash,prompt_version' }
          );
        
        if (ensureErr) {
          console.warn('[REUSE ENSURE ROW FAILED]', ensureErr);
          // Continue anyway - will create row later
        }

        // Now do the reuse check...
        const { data: prev, error: prevErr } = await supabase
          .from('flashcard_generations')
          .select('id, model_name, prompt_version, created_at')
          .eq('user_id', user.id)
          .eq('context_hash', contextHash)
          .eq('prompt_version', promptVersion)
          .is('error_message', null)
          .gte('created_at', sinceIso)
          .order('created_at', { ascending: false })
          .limit(1)
          .maybeSingle();

        if (!prevErr && prev) {
          const { data: prevCards, error: pcErr } = await supabase
            .from('flashcards')
            .select('question, answer, tag, generation_id')
            .eq('generation_id', prev.id);

          if (!pcErr && prevCards?.length) {
            const ageSec = Math.round((Date.now() - new Date(prev.created_at).getTime()) / 1000);
            console.log('[REUSE HIT]', { gen: prev.id, cards: prevCards.length, ageSec });
            return res.json({
              generation_id: prev.id,
              model: prev.model_name,
              requested_count: requestedCount,
              effective_count: requestedCount,
              reused: true,
              cache_age_sec: ageSec,
              flashcards: prevCards,
              prompt_version: prev.prompt_version,
              answer_length: 'mixed',
              applied_tags: [],
              traceId: req.traceId
            });
          }
        }
        console.log('[REUSE MISS]');
      } catch (e) {
        console.warn('[REUSE ERROR]', e);
      }
    } else {
      if (!reuse) console.log('[REUSE SKIPPED] user disabled');
      else console.log('[REUSE BYPASSED] advanced options', {
        answerLength,
        incomingTagsCount: incomingTags.length,
        inferTagsFlag
      });
    }
    mark('reuse_decision');
    // ---------- Ensure generation row (safe UPSERT or fetch existing) ----------
    const supabase = getSupabase();
    const contextHash = sha256Base64(context);
    
    // First check if this exact generation already exists
    const { data: existingGen } = await supabase
      .from('flashcard_generations')
      .select('id')
      .eq('user_id', user.id)
      .eq('context_hash', contextHash)
      .eq('prompt_version', promptVersion)
      .limit(1)
      .maybeSingle();

    if (existingGen) {
      // Use existing generation ID
      generationId = existingGen.id;
      console.log('[REUSING GENERATION]', generationId);
    } else {
      // Create new generation row
      generationId = randomUUID();
      const { data: insertRows, error: insertErr } = await supabase
        .from('flashcard_generations')
        .insert([{
          id: generationId,
          user_id: user.id,
          context_hash: contextHash,
          context_text: context,
          model_name: MODEL_NAME,
          prompt_version: promptVersion,
          created_at: new Date().toISOString(),
        }])
        .select('id')
        .limit(1);

      if (insertErr) {
        console.error('[GEN INSERT ERROR]', insertErr);
        return res.status(500).json({ error: 'Failed to log generation' });
      }
      console.log('[NEW GENERATION]', generationId);
    }
    mark('gen_row');
    // ---------- End ensure generation row ----------

    // ---------- Read user's tag contexts ----------
    let contextFromTags = '';
    try {
      const { data: userTags } = await supabase
        .from('user_tags')
        .select(`
          id,
          tag_name,
          tag_contexts (context_text)
        `)
        .eq('user_id', user.id)
        .order('created_at', { ascending: true });
      
      if (userTags && userTags.length > 0) {
        contextFromTags = 'USER PERSONAL CONTEXT:\n━━━━━━━━━━━━━━━━━━\n';
        for (const tag of userTags) {
          const contexts = (tag.tag_contexts as any[]) || [];
          if (contexts.length > 0) {
            contextFromTags += `\n[${tag.tag_name}]\n`;
            for (const ctx of contexts) {
              contextFromTags += `- ${ctx.context_text}\n`;
            }
          }
        }
        contextFromTags += '\n━━━━━━━━━━━━━━━━━━\n\n';
        console.log('[TAG CONTEXT]', { 
          gen: generationId, 
          tags: userTags.length,
          contextLength: contextFromTags.length 
        });
      }
    } catch (e) {
      console.warn('[TAG CONTEXT ERROR]', e);
      // Continue without tag context if error
    }
    
    // Prepend tag context to user's prompt
    const enhancedContext = contextFromTags + context;

    // ---------- Model call ----------
    // Pass adaptive signal + recentAvg to generator via answerLength + custom distribution meta
    const recentAvg = 0; // Default value or replace with actual logic if needed
    // ...before calling generateFlashcards, gather memory (pseudo; add real retrieval logic):

    const relatedPrompts: string[] = [];
    const previousAnswers: string[] = [];
    const hardBan: string[] = [];
    const softDeprioritize: string[] = [];

    // Decide whether to prefer symbol vocabulary for this prompt.
    // We prefer symbols for need/food contexts and emotion contexts, but now split by type
    const needContextRegex = /\b(eat|eating|hungry|food|breakfast|lunch|dinner|snack|pancake|cereal|sandwich|pizza|drink|thirsty)\b/i;
    const emotionContextRegex = /\b(how are you|how r you|how are u|how r u|how you|how's it going|how is it going|how do you feel)\b/i;
    const hasFoodTag = incomingTags.includes('food');
    const hasFeelingTag = incomingTags.includes('feelings') || incomingTags.includes('emotion');

    const preferFood = hasFoodTag || needContextRegex.test(context);
    const preferEmotions = hasFeelingTag || emotionContextRegex.test(context);
    const preferSymbols = preferFood || preferEmotions;

    // Small rule-based handler for conversational 'how are you' prompts.
    // Rule-based handler removed; always use generateFlashcards
    let genResult: any;
    {
      const { cards, rawContent, modelUsed } = await generateFlashcards(
        enhancedContext,
        requestedCount,
        requestedCount,
        promptVersion,
        answerLength,
        null,
        {
          relatedPrompts,
          previousAnswers,
          hardBan,
          softDeprioritize,
          preferSymbols,
        },
        AVAILABLE_SYMBOLS
      );
      genResult = { cards, rawContent, modelUsed };
    }
    const { cards, rawContent, modelUsed } = genResult;
    mark('model_done');

    // NEW CLEANING BLOCK (replaces old duplicate/schema filtering)
    let rawCards = cards || [];
    // Debug: show raw length and sample
    console.log('[GEN RAW]', {
      gen: generationId,
      raw_count: rawCards.length,
      sample: rawCards.slice(0, 3)
    });

    // Accept only objects with both question & answer strings (case-insensitive fallback)
    rawCards = rawCards
      .map((c: any) => {
        if (c && typeof c === 'object') {
          const q = (c as any).question ?? (c as any).Question;
          const a = (c as any).answer ?? (c as any).Answer;
          const r = (c as any).role ?? 'content';
          const fitz = (c as any).fitz ?? null;
          if (typeof q === 'string' && typeof a === 'string') {
            return { question: q, answer: a, role: r, fitz };
          }
        }
        return null;
      })
      .filter(Boolean) as { question: string; answer: string; role: string; fitz: string | null }[];

    const originalPromptQuestion = context; // ensure we only store the user prompt

    // Clamp answers to 1–3 words & normalize
    function clampShort(ans: string): string {
      return ans
        .replace(/[“”"'.,!?;:]/g, ' ')
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 3)
        .join(' ');
    }

  let deduped: { question: string; answer: string; role: string; fitz: string | null }[] = [];
    const seenAnswers = new Set<string>();

    for (const c of rawCards) {
      const answer = clampShort(c.answer);
      if (!answer) continue;
      if (seenAnswers.has(answer)) continue; // exact duplicate only
      seenAnswers.add(answer);
      deduped.push({
        question: originalPromptQuestion,
        answer,
        role: (c as any).role ?? 'content',
        fitz: (c as any).fitz ?? null,
      });
      if (deduped.length >= requestedCount) break;
    }

    // ----- Post-filter: remove tag-only answers (answers that are exactly tag names)
    try {
      const tagSet = new Set(incomingTags.map(t => t.toLowerCase()));
      const beforeTagFilter = deduped.length;
      if (tagSet.size > 0) {
        deduped = deduped.filter(c => !tagSet.has(c.answer.toLowerCase()));
      }
      const removedTagOnly = beforeTagFilter - deduped.length;
      if (removedTagOnly > 0) {
        console.log('[TAG FILTER] removed tag-only answers', { gen: generationId, removed: removedTagOnly });
      }
    } catch (e) {
      console.warn('[TAG FILTER ERROR]', e);
    }

    // Fallback fill if short
    if (deduped.length < requestedCount) {
      // Always include sentence builders, then pick topic words based on context
      const foodFallback = [
        'pancakes','cake','sandwich','pizza','salad','soup','pasta',
        'fruit','apple','bread','cereal','burger','rice','eggs','water'
      ];
      const emotionFallback = [
        'happy','sad','angry','calm','worried','excited','tired',
        'fine','good','okay','scared','confused','sick','better','worse'
      ];
      const sentenceFallback = [
        'i','am','feel','my','is','not','want','need','have','more','done','help','stop','yes','no'
      ];

      // Always include sentence builders, then pick topic words based on context
      const pool = preferFood
        ? sentenceFallback.concat(foodFallback).concat(emotionFallback)
        : preferEmotions
          ? sentenceFallback.concat(emotionFallback).concat(foodFallback)
          : sentenceFallback.concat(emotionFallback).concat(foodFallback);
      console.log('[FALLBACK POOL]', { gen: generationId, preferSymbols, poolSample: pool.slice(0,6) });
      for (const w of pool) {
        if (deduped.length >= requestedCount) break;
        if (seenAnswers.has(w)) continue;
        seenAnswers.add(w);
        deduped.push({ question: originalPromptQuestion, answer: w, role: 'content', fitz: null });
      }
    }

    // Stats
    const cleanedStats = {
      input: rawCards.length,
      kept: deduped.length,
      rejected_schema: rawCards.length === 0 ? 0 : (rawCards.length - deduped.length),
      rejected_dup_simple: rawCards.length - deduped.length
    };

    let keptCards = deduped;

    console.log('[GEN CLEANED REVISED]', {
      gen: generationId,
      input: cleanedStats.input,
      kept: cleanedStats.kept,
      rejected_dup_simple: cleanedStats.rejected_dup_simple
    });

    // --- Embedding logic (A + E) ---
    let keptEmbeddings: (number[] | null)[] = [];
    let duplicatesRemovedEmbedding = 0;

    const lite =
      req.query.lite === '1' ||
      req.query.lite === 'true' ||
      (rawInput && (rawInput.lite === true || rawInput.lite === 1));

    // New: explicit embed request flag (embed=1 or embed=true required to run synchronous embeddings)
    const embedRequested =
      (req.query.embed === '1' || req.query.embed === 'true' ||
       rawInput.embed === true || rawInput.embed === 1);

    mark('embedding_start');

    if (ENABLE_EMBED && embedRequested && keptCards.length) {
      try {
        // Limit existing embeddings scope (optional: change fetchExistingEmbeddings signature to accept limit)
        const existingEmbeddings = await fetchExistingEmbeddings(user.id /* , { limit: 300 } */);
        keptEmbeddings = await embedBatch(keptCards.map(c => c.question));
        const { keep, removed } = filterDuplicates(keptEmbeddings as number[][], existingEmbeddings);
        keptCards = keptCards.filter((_, i) => keep[i]);
        keptEmbeddings = keptEmbeddings.filter((_, i) => keep[i]);
        duplicatesRemovedEmbedding = removed;
        metrics.duplicatesRemoved += removed;
      } catch (e) {
        console.error('[EMBED FAIL - continuing]', e);
        // Fallback: null embeddings so insert still works
        keptEmbeddings = Array(keptCards.length).fill(null);
      }
    } else {
      // Skip embeddings -> fill nulls
      keptEmbeddings = Array(keptCards.length).fill(null);

      // Deferred (fire-and-forget) embedding if embeddings enabled but not requested
      if (ENABLE_EMBED && keptCards.length && !lite && !embedRequested) {
        (async () => {
          try {
            const vectors = await embedBatch(keptCards.map(c => c.question));
            console.log('[DEFERRED EMBED OK]', { gen: generationId, count: vectors.length });
          } catch (bgErr) {
            console.warn('[DEFERRED EMBED FAIL]', bgErr);
          }
        })();
      }
    }

    mark('embedding_done');

    console.log('[GEN CLEANED]', {
      gen: generationId,
      input: rawCards.length,
      valid: keptCards.length,
      rejected_schema: cleanedStats.rejected_schema,
      rejected_dup_simple: cleanedStats.rejected_dup_simple
    });

    // --- Avoid previous generation duplicates ---
    const avoidGenId = (rawInput.avoid_generation_id || req.query.avoid_generation_id || '').toString().trim() || null;
    if (avoidGenId && /^[0-9a-fA-F-]{36}$/.test(avoidGenId)) {
      try {
        const { data: prevQs } = await getSupabase()
          .from('flashcards')
          .select('question')
          .eq('generation_id', avoidGenId)
          .eq('user_id', user.id);
        if (prevQs?.length) {
            const prevSet = new Set(prevQs.map(r => r.question.toLowerCase()));
            const before = keptCards.length;
            keptCards = keptCards.filter(c => !prevSet.has(c.question.toLowerCase()));
            if (before !== keptCards.length) {
              console.log('[AVOID FILTER]', {
                generationId,
                avoid_generation_id: avoidGenId,
                removed: before - keptCards.length
              });
            }
            // Also update keptEmbeddings to match filtered keptCards
            if (keptEmbeddings.length === before) {
              const filteredEmbeddings: typeof keptEmbeddings = [];
              keptCards.forEach(c => {
                const idx = rawCards.findIndex((rc: { question: string }) => rc.question === c.question);
                filteredEmbeddings.push(keptEmbeddings[idx] || null);
              });
              keptEmbeddings = filteredEmbeddings;
            }
        }
      } catch (e) {
        console.warn('[AVOID FILTER ERROR]', e);
      }
    }
    mark('reuse_decision_end');

    if (!keptCards.length) {
      await logGenerationComplete({
        generationId: generationId ?? '',
        cards: [],
        raw: null,
        latencyMs: Date.now() - t0,
        modelName: modelUsed,
        error: 'no_valid_cards'
      });
      return res.status(422).json({ error: 'No valid cards generated', generation_id: generationId ?? '' });
    }

    // ---------- Answer length (forced short) ----------
    const forcedAnswerLength = 'short'; // sentence-building mode: override any incoming value

    // ---------- Length clamp (short only 1–3 words) ----------
    keptCards = keptCards.map(c => ({
      question: c.question.length > 80 ? c.question.slice(0,77).trimEnd() + '…' : c.question,
      answer: clampShort(c.answer),
      role: c.role ?? 'content',
      fitz: c.fitz ?? null
    }));
    console.log('[POST-CLAMP SHORT]', {
      gen: generationId,
      samples: keptCards.slice(0, 3).map(c => c.answer)
    });

    // ---------- Tags processing ----------
    let finalTagsSet = new Set(incomingTags);
    if (inferTagsFlag) {
      const inferred = inferTagsFromCards(keptCards)
        .filter(t => /^[a-z0-9._-]{3,40}$/.test(t));
      inferred.forEach(t => finalTagsSet.add(t));
    }
    const finalTags = Array.from(finalTagsSet).slice(0, 8);
    console.log('[GEN TAGS]', { gen: generationId, finalTags });

    // ---------- Insert flashcards (B + E) ----------
    const rows = keptCards.map((c, i) => {
      // Match the answer to an actual symbol filename
      const symbolFile = matchSymbolFilename(c.answer, AVAILABLE_SYMBOLS);
      return {
        generation_id: generationId,
        user_id: user.id,
        question: c.question,
        answer: c.answer,
        asset_filename: symbolFile, // Use matched filename
        tag: tag || null,
        embedding: keptEmbeddings[i],
        created_at: new Date().toISOString()
      };
    });

    console.log('[GEN PRE-INSERT]', {
      gen: generationId,
      rows: rows.length,
      firstRow: rows[0]
    });

    mark('db_insert_start');
    const { data: inserted, error: insertErr } = await getSupabase()
      .from('flashcards')
      .insert(rows)
      .select('id, question, answer, tag, asset_filename'); // ✅ SELECT asset_filename from database
    mark('db_insert_done');

    if (insertErr) {
      console.error('[flashcards insert error]', {
        gen: generationId,
        code: insertErr.code,
        message: insertErr.message,
        details: insertErr.details
      });
      return res.status(500).json({ error: 'DB insert failed', generation_id: generationId });
    }
    console.log('[flashcards insert ok]', { generationId, inserted: inserted?.length || 0 });

    // ✅ Use stored asset_filename from database (already matched during insert)
    const responseCards = inserted.map((card: any, i: number) => ({
      id: card.id,
      question: card.question,
      answer: card.answer,
      tags: finalTags,
      asset_filename: card.asset_filename || 'blank.svg', // Use stored value from DB
      fitz: keptCards[i]?.fitz ?? null,
    }));

    // ---------- Tags insert (reuse inserted list) ----------
    if (finalTags.length && inserted?.length) {
      console.log('[TAG INSERT]', { gen: generationId, tagCount: finalTags.length });
      const tagRows = [];
      for (const card of inserted) {
        for (const tg of finalTags) {
          tagRows.push({ card_id: card.id, tag: tg });
        }
      }
      if (tagRows.length) {
        const { error: tagErr } = await getSupabase()
          .from('flashcard_tags')
          .insert(tagRows);
        if (tagErr) console.error('[tag insert error]', tagErr);
      }
    }

    // ---------- Log completion ----------    
    try {
      await logGenerationComplete({
        generationId: generationId ?? '',
        cards: keptCards,
        raw: null,
        latencyMs: Date.now() - t0,
        modelName: modelUsed
      });
    } catch (e) {
      console.error('[logGenerationComplete failed]', e);
    }

    metrics.generationSuccess++;
    metrics.totalLatencyMs += Date.now() - t0;

    return res.json({
      generation_id: generationId,
      requested_count: requestedCount,
      effective_count: effectiveCount,
      reused: false,
      reused_generation: reusedGenerationRecord,   // <--- ADD THIS
      answer_length: forcedAnswerLength,
      applied_tags: finalTags,
      lite,
      flashcards: responseCards,
      timings: stageTimes
    });
  } catch (e: any) {
    metrics.generationError++;
    metrics.totalLatencyMs += Date.now() - t0;
    console.error('[GEN FATAL]', e);
    if (generationId) {
      try {
        await logGenerationComplete({
          generationId,
          cards: [],
          raw: null,
          latencyMs: Date.now() - t0,
          modelName: modelUsed,
          error: e?.message || 'error'
        });
      } catch {}
    }
    const isClient = /Invalid body|context exceeds/.test(e?.message || '');
    res.status(isClient ? 400 : 500).json({
      error: e?.message || 'Server error',
      generation_id: generationId,
      traceId: req.traceId
    });
  }
});

// --- FAVORITES ROUTES (canonical) ---
app.get('/flashcards/favorites', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });

  try {
    const sb = getSupabase();
    const { data: favRows, error: favErr } = await sb
      .from('flashcard_favorites')
      .select('card_id, created_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: true });
    if (favErr) throw favErr;

    const ids = (favRows || []).map((r: any) => r.card_id);
    if (!ids.length) return res.json({ favorites: [], count: 0 });

    //  SELECT asset_filename from database
    const { data: cards, error: cardErr } = await sb
      .from('flashcards')
      .select('id, question, answer, asset_filename, tag, generation_id, created_at')
      .in('id', ids)
      .eq('user_id', user.id);
    if (cardErr) throw cardErr;

    const order = new Map(ids.map((id: string, i: number) => [id, i]));
    (cards || []).sort((a: any, b: any) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));

    //  Ensure asset_filename exists, fallback to blank.svg
    const enrichedCards = (cards || []).map((card: any) => ({
      ...card,
      asset_filename: card.asset_filename || 'blank.svg'
    }));

    return res.json({ favorites: enrichedCards, count: enrichedCards.length });
  } catch (e) {
    console.error('[FAVORITES LIST ERROR]', e);
    return res.status(500).json({ error: 'favorites_list_failed' });
  }
});

app.post('/flashcards/:id/favorite', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  const cardId = req.params.id;

  try {
    const sb = getSupabase();
    
    // Ensure user exists in users table (upsert to avoid foreign key constraint)
    console.log('[FAVORITE] Ensuring user exists:', { userId: user.id, email: user.email });
    
    // First, check what columns exist in users table
    const { data: existingUser } = await sb.from('users').select('*').eq('id', user.id).single();
    
    if (existingUser) {
      console.log('[USER EXISTS]', { userId: user.id });
    } else {
      // User doesn't exist, need to create with all required NOT NULL fields
      const { error: userUpsertErr } = await sb.from('users').upsert([{ 
        id: user.id, 
        email: user.email || `${user.id}@placeholder.local`,
        name: 'Anonymous', // Add name field (required NOT NULL)
        role: 'user' // Add role field (required NOT NULL)
      }], { onConflict: 'id' });
      
      if (userUpsertErr) {
        console.error('[USER UPSERT ERROR]', userUpsertErr);
        // Continue anyway - try the favorite insert
      } else {
        console.log('[USER UPSERT OK]', { userId: user.id });
      }
    }
    
    const { data: card, error: cErr } = await sb
      .from('flashcards').select('id,user_id').eq('id', cardId).single();
    if (cErr || !card || card.user_id !== user.id) {
      return res.status(404).json({ error: 'flashcard_not_found' });
    }
    const { error: fErr } = await sb
      .from('flashcard_favorites')
      .upsert([{ user_id: user.id, card_id: cardId }], { onConflict: 'user_id,card_id' });
    if (fErr) throw fErr;

    res.json({ ok: true, favorite: true });
  } catch (e) {
    console.error('[FAVORITE ADD ERROR]', e);
    res.status(500).json({ error: 'favorite_add_failed' });
  }
});

app.delete('/flashcards/:id/favorite', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  const cardId = req.params.id;

  try {
    const { error } = await getSupabase()
      .from('flashcard_favorites')
      .delete()
      .eq('user_id', user.id)
      .eq('card_id', cardId);
    if (error) throw error;

    res.json({ ok: true });
  } catch (e) {
    console.error('[FAVORITE REMOVE ERROR]', e);
    res.status(500).json({ error: 'favorite_remove_failed' });
  }
});

// ============================================
// OPTIMIZATION SCREEN API ROUTES
// ============================================

// --- TAG MANAGEMENT ---
app.get('/optimization/tags', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  try {
    const { data: tags, error } = await getSupabase()
      .from('user_tags')
      .select('id, tag_name, created_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: true });
      
    if (error) throw error;
    res.json({ tags: tags || [] });
  } catch (e) {
    console.error('[OPTIMIZATION TAGS GET ERROR]', e);
    res.status(500).json({ error: 'failed_to_load_tags' });
  }
});

// --- Seed default favorites for a user (idempotent) ---
// Refactor seeding logic into a reusable helper so webhook or routes can call it
async function createDefaultFavoritesForUser(sb: any, user: { id: string; email?: string | null }) {
  const defaults = [
    'yes','no','more','help','eat','drink','happy','sad','tired','sleep','home','play',
    'go','stop','please','thank you','i','you','want','like','all done','again'
  ];

  // Ensure user exists in users table (best-effort)
  try {
    const { data: existing } = await sb.from('users').select('id').eq('id', user.id).single();
    if (!existing) {
      await sb.from('users').upsert([{
        id: user.id,
        email: user.email || `${user.id}@placeholder.local`,
        name: user.email ? user.email.split('@')[0] : 'Anonymous',
        role: 'user'
      }], { onConflict: 'id' });
    }
  } catch (e) {
    // ignore upsert errors
  }

  const createdFavorites: any[] = [];
  for (const word of defaults) {
    // find existing flashcard for this user with this answer
    const { data: existingCards } = await sb.from('flashcards')
      .select('id, asset_filename')
      .eq('user_id', user.id)
      .ilike('answer', word)
      .limit(1);

    let cardId: string | null = null;
    let assetFilename: string | null = null;
    if (existingCards && existingCards.length) {
      cardId = existingCards[0].id;
      assetFilename = existingCards[0].asset_filename || null;
    } else {
      // create a minimal flashcard row
      const generationId = randomUUID();
      const symbol = matchSymbolFilename(word, AVAILABLE_SYMBOLS) || 'blank.svg';

      // Ensure a corresponding flashcard_generation exists to satisfy FK
      const ctxHash = sha256Base64(word);
      const { data: genUpsert, error: genUpsertErr } = await sb.from('flashcard_generations').upsert([{
        id: generationId,
        user_id: user.id,
        context_hash: ctxHash,
        context_text: word,
        model_name: 'builtin-defaults',
        prompt_version: 1,
        created_at: new Date().toISOString()
      }], { onConflict: 'id' });
      if (genUpsertErr) {
        console.error('[SEED DEFAULTS] generation upsert failed', genUpsertErr);
        throw genUpsertErr;
      }

      const { data: inserted, error: insertErr } = await sb.from('flashcards')
        .insert([{
          generation_id: generationId,
          user_id: user.id,
          question: word,
          answer: word,
          asset_filename: symbol,
          tag: null,
          created_at: new Date().toISOString()
        }])
        .select('id, asset_filename')
        .limit(1);
      if (insertErr) {
        console.error('[SEED DEFAULTS] insert flashcard error', insertErr);
        throw insertErr;
      } else if (inserted && inserted.length) {
        cardId = inserted[0].id;
        assetFilename = inserted[0].asset_filename || null;
      }
    }

    if (cardId) {
      // upsert favorite
      const { error: favErr } = await sb.from('flashcard_favorites')
        .upsert([{ user_id: user.id, card_id: cardId }], { onConflict: 'user_id,card_id' });
      if (!favErr) {
        createdFavorites.push({ card_id: cardId, answer: word, asset_filename: assetFilename });
      }
    }
  }

  return createdFavorites;
}

app.post('/users/seed-default-favorites', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  try {
    const sb = getSupabase();
    const created = await createDefaultFavoritesForUser(sb, user);
    return res.json({ created, count: created.length });
  } catch (e) {
    console.error('[SEED DEFAULTS ERROR]', e);
    return res.status(500).json({ error: 'seed_failed' });
  }
});

// Webhook endpoint for Supabase auth / external systems to notify of new users.
// Configure your Supabase project to POST to this endpoint on user creation.
app.post('/webhooks/supabase', async (req, res) => {
  // Optional shared secret header: set WEBHOOK_SECRET in server env to require it
  const secret = process.env.WEBHOOK_SECRET;
  if (secret) {
    const header = (req.headers['x-webhook-secret'] || req.headers['x-supabase-signature']) as string | undefined;
    if (!header || header !== secret) {
      console.warn('[WEBHOOK] invalid secret header');
      return res.status(403).json({ error: 'forbidden' });
    }
  }

  try {
    const payload = req.body || {};

    // Attempt to find user id across common webhook payload shapes
    const userObj = payload.user || payload.record || payload.new || payload.data || payload;
    const userId = userObj?.id || userObj?.user?.id || payload?.user_id || payload?.meta?.user_id;
    const email = userObj?.email || (userObj?.user && userObj.user.email) || null;

    if (!userId) return res.status(400).json({ error: 'no_user_id_provided' });

    const sb = getSupabase();
    const created = await createDefaultFavoritesForUser(sb, { id: userId, email });
    console.log('[WEBHOOK] seeded defaults for user', userId, 'created_count', created.length);
    return res.json({ ok: true, created_count: created.length });
  } catch (e) {
    console.error('[WEBHOOK SEED ERROR]', e);
    return res.status(500).json({ error: 'webhook_failed' });
  }
});

app.post('/optimization/tags', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const { tag_name } = req.body;
  if (!tag_name || typeof tag_name !== 'string') {
    return res.status(400).json({ error: 'tag_name required' });
  }
  
  try {
    const { data, error } = await getSupabase()
      .from('user_tags')
      .insert([{ user_id: user.id, tag_name: tag_name.trim() }])
      .select()
      .single();
      
    if (error) throw error;
    res.json({ tag: data });
  } catch (e: any) {
    if (e?.code === '23505') { // unique constraint violation
      return res.status(409).json({ error: 'tag_already_exists' });
    }
    console.error('[OPTIMIZATION TAG CREATE ERROR]', e);
    res.status(500).json({ error: 'failed_to_create_tag' });
  }
});

app.delete('/optimization/tags/:id', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const tagId = req.params.id;
  
  try {
    // Verify tag belongs to user before deleting
    const { data: tag } = await getSupabase()
      .from('user_tags')
      .select('id')
      .eq('id', tagId)
      .eq('user_id', user.id)
      .single();
      
    if (!tag) {
      return res.status(404).json({ error: 'tag_not_found' });
    }
    
    const { error } = await getSupabase()
      .from('user_tags')
      .delete()
      .eq('id', tagId);
      
    if (error) throw error;
    res.json({ ok: true });
  } catch (e) {
    console.error('[OPTIMIZATION TAG DELETE ERROR]', e);
    res.status(500).json({ error: 'failed_to_delete_tag' });
  }
});

// --- CONTEXT MANAGEMENT ---
app.get('/optimization/tags/:tagId/contexts', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const tagId = req.params.tagId;
  
  try {
    // Verify tag belongs to user
    const { data: tag } = await getSupabase()
      .from('user_tags')
      .select('id')
      .eq('id', tagId)
      .eq('user_id', user.id)
      .single();
      
    if (!tag) {
      return res.status(404).json({ error: 'tag_not_found' });
    }
    
    const { data: contexts, error } = await getSupabase()
      .from('tag_contexts')
      .select('id, context_text, created_at')
      .eq('tag_id', tagId)
      .order('created_at', { ascending: true });
      
    if (error) throw error;
    res.json({ contexts: contexts || [] });
  } catch (e) {
    console.error('[OPTIMIZATION CONTEXTS GET ERROR]', e);
    res.status(500).json({ error: 'failed_to_load_contexts' });
  }
});

app.post('/optimization/tags/:tagId/contexts', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const tagId = req.params.tagId;
  const { context_text } = req.body;
  
  if (!context_text || typeof context_text !== 'string') {
    return res.status(400).json({ error: 'context_text required' });
  }
  
  try {
    // Verify tag belongs to user
    const { data: tag } = await getSupabase()
      .from('user_tags')
      .select('id')
      .eq('id', tagId)
      .eq('user_id', user.id)
      .single();
      
    if (!tag) {
      return res.status(404).json({ error: 'tag_not_found' });
    }
    
    const { data, error } = await getSupabase()
      .from('tag_contexts')
      .insert([{ tag_id: tagId, context_text: context_text.trim() }])
      .select()
      .single();
      
    if (error) throw error;
    res.json({ context: data });
  } catch (e) {
    console.error('[OPTIMIZATION CONTEXT CREATE ERROR]', e);
    res.status(500).json({ error: 'failed_to_create_context' });
  }
});

app.delete('/optimization/contexts/:id', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const contextId = req.params.id;
  
  try {
    // Verify context belongs to user's tag
    const { data: context } = await getSupabase()
      .from('tag_contexts')
      .select('id, tag_id, user_tags!inner(user_id)')
      .eq('id', contextId)
      .single();
      
    if (!context || (context.user_tags as any).user_id !== user.id) {
      return res.status(404).json({ error: 'context_not_found' });
    }
    
    const { error } = await getSupabase()
      .from('tag_contexts')
      .delete()
      .eq('id', contextId);
      
    if (error) throw error;
    res.json({ ok: true });
  } catch (e) {
    console.error('[OPTIMIZATION CONTEXT DELETE ERROR]', e);
    res.status(500).json({ error: 'failed_to_delete_context' });
  }
});

// Allow editing an existing tag context sentence
app.patch('/optimization/contexts/:id', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  const contextId = req.params.id;
  const { context_text } = req.body;
  if (!context_text || typeof context_text !== 'string') {
    return res.status(400).json({ error: 'context_text required' });
  }

  try {
    console.log('[OPT CONTEXT UPDATE] user=', user?.id, 'contextId=', contextId, 'body_present=', !!req.body);
    // Verify ownership: join tag_contexts -> user_tags -> user_id
    const { data: ctxRow, error: ctxErr } = await getSupabase()
      .from('tag_contexts')
      .select('id, tag_id')
      .eq('id', contextId)
      .limit(1)
      .maybeSingle();
    if (ctxErr) throw ctxErr;
    if (!ctxRow) return res.status(404).json({ error: 'context_not_found' });

    const { data: tagRow, error: tagErr } = await getSupabase()
      .from('user_tags')
      .select('id, user_id')
      .eq('id', ctxRow.tag_id)
      .limit(1)
      .maybeSingle();
    if (tagErr) throw tagErr;
    if (!tagRow || tagRow.user_id !== user.id) return res.status(404).json({ error: 'context_not_found' });

    const { data, error } = await getSupabase()
      .from('tag_contexts')
      .update({ context_text: context_text.trim() })
      .eq('id', contextId)
      .select()
      .single();
    if (error) {
      console.error('[OPTIMIZATION CONTEXT UPDATE ERROR]', error);
      throw error;
    }
    if (!data) return res.status(500).json({ error: 'update_failed' });
    res.json({ context: data });
  } catch (e) {
    console.error('[OPTIMIZATION CONTEXT UPDATE ERROR]', e);
    res.status(500).json({ error: 'failed_to_update_context' });
  }
});

// --- FAVORITES MANAGEMENT ---
app.get('/optimization/favorites', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  try {
    const sb = getSupabase();

    // 1) Fetch optimization UI favorites (user_favorites)
    const { data: optFavs = [], error: optErr } = await sb
      .from('user_favorites')
      .select('id, word, asset_filename, created_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: true });
    if (optErr) throw optErr;

    // 2) Fetch canonical favorites (flashcard_favorites joined to flashcards)
    const { data: canonFavs = [], error: canonErr } = await sb
      .from('flashcard_favorites')
      .select('card_id, created_at, flashcards(id, answer, asset_filename)')
      .eq('user_id', user.id)
      .order('created_at', { ascending: true });
    if (canonErr) throw canonErr;

    // Debug: log counts so we can see if seeded/canonical favorites exist for this user
    try {
      console.log('[OPTIMIZATION/FAVORITES] user=', user.id, 'optFavs_count=', (optFavs || []).length, 'canonFavs_count=', (canonFavs || []).length);
      if (canonFavs && canonFavs.length) {
        const canonPreview = (canonFavs as any[]).slice(0,10).map(r => ({ card_id: r.card_id, answer: r.flashcards?.answer }));
        console.log('[OPTIMIZATION/FAVORITES] canonical preview=', JSON.stringify(canonPreview));
      }
    } catch (e) {
      // silent
    }

    // Normalize canonical results to same shape as optimization favorites
    const canonMapped = (canonFavs || []).map((r: any) => {
      const card = r.flashcards || r.flashcard || {}; // defensive
      return {
        id: `card_${r.card_id}`,
        word: (card.answer || '').toString(),
        asset_filename: card.asset_filename || null,
        created_at: r.created_at
      };
    });

    // Merge, prefer canonical flashcards when duplicate words exist (case-insensitive)
    const merged: any[] = [];
    const seen = new Map<string, number>(); // normalized word -> index

    const pushIfNew = (item: any) => {
      const key = (item.word || '').toString().trim().toLowerCase();
      if (!key) return;
      if (seen.has(key)) return; // already added
      seen.set(key, merged.length);
      merged.push(item);
    };

    // First add canonical favorites so they take precedence
    for (const c of canonMapped) pushIfNew(c);
    // Then add optimization-only favorites
    for (const o of (optFavs || [])) pushIfNew(o);

    // Sort by created_at ascending (keep canonical relative order first)
    merged.sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

    res.json({ favorites: merged });
  } catch (e) {
    console.error('[OPTIMIZATION FAVORITES GET ERROR]', e);
    res.status(500).json({ error: 'failed_to_load_favorites' });
  }
});

app.post('/optimization/favorites', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  const { word, asset_filename } = req.body;
  if (!word || typeof word !== 'string') {
    return res.status(400).json({ error: 'word required' });
  }
  
  try {
    const sb = getSupabase();

    // Insert into optimization table (user_favorites)
    const { data: ufData, error: ufErr } = await sb
      .from('user_favorites')
      .insert([{ 
        user_id: user.id, 
        word: word.trim(),
        asset_filename: asset_filename || null
      }])
      .select()
      .maybeSingle();

    if (ufErr) {
      // If unique constraint, continue to attempt sync with existing record
      if (ufErr.code !== '23505') throw ufErr;
    }

    // --- Sync into canonical flashcards + favorites ---
    // 1) Try to find an existing flashcard for this user with the same answer
    const trimmed = word.trim();
    const { data: existingCards } = await sb.from('flashcards')
      .select('id, asset_filename')
      .eq('user_id', user.id)
      .ilike('answer', trimmed)
      .limit(1);

    let cardId: string | null = null;
    let chosenAsset: string | null = asset_filename || null;

    if (existingCards && existingCards.length) {
      cardId = existingCards[0].id;
      chosenAsset = chosenAsset || existingCards[0].asset_filename || null;
    } else {
      // Create flashcard row and pick an asset via symbol matching if not provided
      const generationId = randomUUID();
      const symbol = chosenAsset || matchSymbolFilename(trimmed, AVAILABLE_SYMBOLS) || 'blank.svg';

      // Ensure flashcard_generations row exists for FK
      const ctxHash = sha256Base64(trimmed);
      const { data: genUpsert2, error: genUpsertErr2 } = await sb.from('flashcard_generations').upsert([{
        id: generationId,
        user_id: user.id,
        context_hash: ctxHash,
        context_text: trimmed,
        model_name: 'optimization-fav',
        prompt_version: 1,
        created_at: new Date().toISOString()
      }], { onConflict: 'id' });
      if (genUpsertErr2) {
        console.error('[OPTIMIZATION SYNC] generation upsert failed', genUpsertErr2);
        throw genUpsertErr2;
      }

      const { data: inserted, error: insertErr } = await sb.from('flashcards')
        .insert([{
          generation_id: generationId,
          user_id: user.id,
          question: trimmed,
          answer: trimmed,
          asset_filename: symbol,
          tag: null,
          created_at: new Date().toISOString()
        }])
        .select('id, asset_filename')
        .limit(1);
      if (insertErr) {
        console.error('[OPTIMIZATION SYNC] flashcard insert error', insertErr);
        throw insertErr;
      } else if (inserted && inserted.length) {
        cardId = inserted[0].id;
        chosenAsset = inserted[0].asset_filename || chosenAsset;
      }
    }

    // 2) Upsert into flashcard_favorites so the Home favorites show it
    if (cardId) {
      const { error: favErr } = await sb.from('flashcard_favorites')
        .upsert([{ user_id: user.id, card_id: cardId }], { onConflict: 'user_id,card_id' });
      if (favErr) console.error('[OPTIMIZATION SYNC] flashcard_favorites upsert error', favErr);
    }

    // 3) Return merged favorites (same shape as GET /optimization/favorites) so client sees immediate update
    try {
      const { data: optFavs = [] } = await sb
        .from('user_favorites')
        .select('id, word, asset_filename, created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: true });

      const { data: canonFavs = [] } = await sb
        .from('flashcard_favorites')
        .select('card_id, created_at, flashcards(id, answer, asset_filename)')
        .eq('user_id', user.id)
        .order('created_at', { ascending: true });

      const canonMapped = (canonFavs || []).map((r: any) => {
        const card = r.flashcards || r.flashcard || {};
        return {
          id: `${card.id}`,
          word: (card.answer || '').toString(),
          asset_filename: card.asset_filename || null,
          created_at: r.created_at
        };
      });

      const merged: any[] = [];
      const seen = new Set<string>();
      const pushIfNew = (item: any) => {
        const key = (item.word || '').toString().trim().toLowerCase();
        if (!key) return;
        if (seen.has(key)) return;
        seen.add(key);
        merged.push(item);
      };
      for (const c of canonMapped) pushIfNew(c);
      for (const o of (optFavs || [])) pushIfNew(o);
      merged.sort((a: any, b: any) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

      return res.json({
        optimization_favorite: ufData || null,
        flashcard_id: cardId,
        asset_filename: chosenAsset || 'blank.svg',
        favorites: merged
      });
    } catch (e) {
      console.error('[OPTIMIZATION FAVORITE POST - FETCH MERGE ERROR]', e);
      return res.json({ optimization_favorite: ufData || null, flashcard_id: cardId, asset_filename: chosenAsset || 'blank.svg' });
    }
  } catch (e: any) {
    if (e?.code === '23505') { // unique constraint
      return res.status(409).json({ error: 'favorite_already_exists' });
    }
    console.error('[OPTIMIZATION FAVORITE CREATE ERROR]', e);
    res.status(500).json({ error: 'failed_to_create_favorite' });
  }
});

app.delete('/optimization/favorites/:id', async (req, res) => {
  const user = (req as any).user;
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  
  let favoriteId = req.params.id;
  
  try {
    // Handle canonical favorite IDs that have "card_" prefix
    if (favoriteId.startsWith('card_')) {
      const cardId = favoriteId.replace('card_', '');
      // Delete from flashcard_favorites table (canonical)
      const { error } = await getSupabase()
        .from('flashcard_favorites')
        .delete()
        .eq('card_id', cardId)
        .eq('user_id', user.id);
      if (error) throw error;
    } else {
      // Delete from user_favorites table (optimization-only)
      const { error } = await getSupabase()
        .from('user_favorites')
        .delete()
        .eq('id', favoriteId)
        .eq('user_id', user.id);
      if (error) throw error;
    }
    
    res.json({ ok: true });
  } catch (e) {
    console.error('[OPTIMIZATION FAVORITE DELETE ERROR]', e);
    res.status(500).json({ error: 'failed_to_delete_favorite' });
  }
});

// Add this helper function near the top (after loadAvailableSymbols):

function matchSymbolFilename(answer: string, availableSymbols: string[]): string {
  const normalized = answer.toLowerCase().trim().replace(/\s+/g, '_');
  
  // 1. Exact match with .svg extension
  const exactWithExt = availableSymbols.find(s => 
    s.toLowerCase() === `${normalized}.svg`
  );
  if (exactWithExt) {
    console.log('[EXACT MATCH]', { answer, matched: exactWithExt });
    return exactWithExt;
  }
  
  // 2. Exact match without extension
  const exactDirect = availableSymbols.find(s =>
    s.toLowerCase() === normalized || s.toLowerCase() === `${normalized}.svg`
  );
  if (exactDirect) {
    console.log('[DIRECT MATCH]', { answer, matched: exactDirect });
    return exactDirect;
  }
  
  // 3. ✅ STRICT partial prefix match - must be at START of word boundary
  const partial = availableSymbols.find(s => {
    const lower = s.toLowerCase();
    // Must start with the word followed by underscore or .svg
    return lower === `${normalized}_` || 
           lower.startsWith(`${normalized}_`) ||
           lower === `${normalized}.svg`;
  });
  if (partial) {
    console.log('[PARTIAL MATCH]', { answer, matched: partial });
    return partial;
  }
  
  // 4. 🧠 Centralized Synonym Mapping - cover all common AAC vocabulary
   const synonymMap: Record<string, string> = {
    // === Emotions ===
    'happy': 'happy_lady',
    'glad': 'happy_lady',
    'joyful': 'happy_lady',
    'cheerful': 'happy_lady',
    'sad': 'sad_lady',
    'unhappy': 'sad_lady',
    'upset': 'sad_lady',
    'miserable': 'sad_lady',
    'angry': 'angry_lady',
    'mad': 'angry_lady',
    'frustrated': 'angry_lady',
    'cross': 'angry_lady',
    'tired': 'yawn_,_to',
    'exhausted': 'yawn_,_to',
    'sleepy': 'yawn_,_to',
    'drowsy': 'yawn_,_to',
    'sick': 'vomit_,_to',
    'ill': 'vomit_,_to',
    'unwell': 'vomit_,_to',
    'scared': 'afraid_lady',
    'afraid': 'afraid_lady',
    'frightened': 'afraid_lady',
    'terrified': 'afraid_lady',

    // === Actions ===
    'eat': 'eat_,_to',
    'eating': 'eat_,_to',
    'drink': 'drink_,_to',
    'drinking': 'drink_,_to',
    'help': 'help_,_to',
    'want': 'want_,_to',
    'go': 'go_through_door_,_to',
    'run': 'run_,_to',
    'running': 'run_,_to',
    'wait': 'wait_,_to',
    'waiting': 'wait_,_to',
    'sleep': 'sleep_female_,_to',
    'sleeping': 'sleep_female_,_to',
    'nap': 'sleep_female_,_to',
    'play': 'play_area',
    'playing': 'play_area',
    'finish': 'finish',
    'done': 'finish',
    'finished': 'finish',
    'complete': 'finish',

    // === Social / Responses ===
    'yes': 'correct',
    'yeah': 'correct',
    'yep': 'correct',
    'ok': 'correct',
    'okay': 'correct',
    'no': 'mistake_no_wrong',
    'nope': 'mistake_no_wrong',
    'nah': 'mistake_no_wrong',
    'more': 'more',
    'again': 'more',

    // === Sensory ===
    'cold': 'drink_cold',
    'cool': 'drink_cold',
    'hot': 'drink_hot',
    'warm': 'drink_hot',

    // === People ===
    'i': 'I',
    'mom': 'mum_parent',
    'mum': 'mum_parent',
    'mother': 'mum_parent',
    'mommy': 'mum_parent',
    'mama': 'mum_parent',
    'dad': 'dad_parent',
    'father': 'dad_parent',
    'daddy': 'dad_parent',
    'papa': 'dad_parent',

    // === Places ===
    'home': 'motor_home',
    'house': 'motor_home',
    'school': 'school',

    // === Food & Drink ===
    'water': 'water',
    'food': 'cat_food',
    'meal': 'cat_food',
    'breakfast': 'breakfast',
    'lunch': 'lunch',
    'dinner': 'dinner',
    'snack': 'snack',
    'hungry': 'hungry',
    'thirsty': 'water',
    'pancake': 'pancake',
    'pancakes': 'pancake',
    'sandwich': 'sandwich',
    'pizza': 'pizza',
    'apple': 'apple',
    'cereal': 'cereal',
    'bread': 'bread',
    'burger': 'burger',

    // === Time ===
    'morning': 'morning',
    'night': 'moon',
    'bedtime': 'bed_time',
  };
  
  const synonym = synonymMap[normalized];
  if (synonym) {
    const synonymMatch = availableSymbols.find(s => 
      s.toLowerCase() === `${synonym}.svg` || 
      s.toLowerCase().startsWith(`${synonym}_`)
    );
    if (synonymMatch) {
      console.log('[SYNONYM MATCH]', { answer, synonym, matched: synonymMatch });
      return synonymMatch;
    }
  }

  // 4.5 Token-level exact/synonym match: if answer is multiword, try each token
  if (normalized.includes(' ')) {
    const tokens = normalized.split(/\s+/).map(t => t.trim()).filter(Boolean);
    for (const t of tokens) {
      // exact token match
      const exactTok = availableSymbols.find(s => {
        const base = s.toLowerCase().replace('.svg', '');
        return base === t || s.toLowerCase() === `${t}.svg`;
      });
      if (exactTok) {
        console.log('[TOKEN EXACT MATCH]', { token: t, matched: exactTok });
        return exactTok;
      }
      // synonym for token
      const tokSyn = synonymMap[t];
      if (tokSyn) {
        const synMatch = availableSymbols.find(s => s.toLowerCase() === `${tokSyn}.svg` || s.toLowerCase().startsWith(`${tokSyn}_`));
        if (synMatch) {
          console.log('[TOKEN SYNONYM MATCH]', { token: t, syn: tokSyn, matched: synMatch });
          return synMatch;
        }
      }
    }
  }

  // New: direct token map lookup (single-word normalized answers)
  if (!normalized.includes(' ')) {
    const tokenList = SYMBOL_TOKEN_MAP.get(normalized);
    if (tokenList && tokenList.length) {
      // Prefer exact single-token filenames (where base === normalized)
      const exactToken = tokenList.find(s => s.toLowerCase().replace('.svg','') === normalized);
      if (exactToken) {
        console.log('[TOKEN MAP EXACT]', { answer, matched: exactToken });
        return exactToken;
      }
      // Otherwise prefer short filenames where the first token equals normalized
      const pref = tokenList.find(s => s.toLowerCase().split('_')[0] === normalized);
      if (pref) {
        console.log('[TOKEN MAP PREFIX]', { answer, matched: pref });
        return pref;
      }
      // fallback to first token match
      console.log('[TOKEN MAP CHOICE]', { answer, matched: tokenList[0] });
      return tokenList[0];
    }
    // For short words, try synonym map before giving up
    if (normalized.length < 4) {
      const shortSyn = synonymMap[normalized];
      if (shortSyn) {
        const shortMatch = availableSymbols.find(s =>
          s.toLowerCase() === `${shortSyn}.svg` ||
          s.toLowerCase().startsWith(`${shortSyn}_`)
        );
        if (shortMatch) {
          console.log('[SHORT SYNONYM MATCH]', { answer, synonym: shortSyn, matched: shortMatch });
          return shortMatch;
        }
      }
      console.log('[SHORT SINGLE TOKEN - NO MATCH]', { answer, normalized });
      return 'blank.svg';
    }
  }
  
  // 5. Safe Fuzzy Match with Constraints
  // Only allow fuzzy matching for words length >= 4 to avoid short-word collisions
  if (normalized.length >= 4) {
    let bestMatch: string | null = null;
    let bestDistance = 99;
    for (const symbol of availableSymbols) {
      const base = symbol.toLowerCase().replace('.svg', '').split('_')[0]; // first word only

      // Must share same starting letter
      if (!base || base[0] !== normalized[0]) continue;

      // Require base symbol to be reasonably close in length
      if (Math.abs(base.length - normalized.length) > 3) continue;

      const dist = levenshteinDistance(normalized, base);
      if (dist < bestDistance) {
        bestDistance = dist;
        bestMatch = symbol;
      }
    }

    // Acceptance: allow only distance 1 for reasonable matches
    const acceptDist = 1;
    if (bestMatch && bestDistance <= acceptDist) {
      console.log('[FUZZY MATCH]', { answer, distance: bestDistance, matched: bestMatch });
      return bestMatch;
    }
  }
  
  // 5.5 STRONGER CONTAINS-BASED MATCH: accept only when token boundaries or close similarity
  // Avoid arbitrary substring matches that can pick unrelated assets.
  const containsCandidates: string[] = [];
  for (const s of availableSymbols) {
    const base = s.toLowerCase().replace('.svg', '');
    const parts = base.split('_').map(p => p.trim()).filter(Boolean);
    // Accept if any token equals normalized (handled earlier), or
    // if any token startsWith normalized and length difference small
    for (const p of parts) {
      if (p === normalized) {
        containsCandidates.push(s);
        break;
      }
      if (p.startsWith(normalized) && Math.abs(p.length - normalized.length) <= 3 && normalized.length >= 3) {
        containsCandidates.push(s);
        break;
      }
      // allow if normalized is contained and lengths are similar (avoid tiny substrings)
      if (p.includes(normalized) && Math.abs(p.length - normalized.length) <= 2 && normalized.length >= 4) {
        containsCandidates.push(s);
        break;
      }
    }
  }

  if (containsCandidates.length) {
    containsCandidates.sort((a,b) => a.length - b.length);
    console.log('[CONSTRAINED CONTAINS MATCH]', { answer, matched: containsCandidates[0], candidates: containsCandidates.slice(0,5) });
    return containsCandidates[0];
  }
  
  console.warn('[NO MATCH]', { answer, using_fallback: 'blank.svg' });
  try {
    const logPath = path.resolve(__dirname, '../../unmatched.txt');
    fs.appendFileSync(logPath, `${answer}\n`, 'utf-8');
  } catch (e) {}
  return 'blank.svg';
}

function levenshteinDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const matrix: number[][] = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) matrix[i][j] = matrix[i - 1][j - 1];
      else matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1);
    }
  }
  return matrix[b.length][a.length];
}


app.use((req, res) => {
  res.status(404).json({ error: 'not_found' });
});

app.listen(PORT, () => {
  console.log(`[LISTEN] Server running on port ${PORT}`);
});



