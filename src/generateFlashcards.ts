import OpenAI from 'openai';

export interface Flashcard {
  question: string;
  answer: string;
  role?: CardRole; // expose role so UI can style/group cards
  fitz?: FitzCategory | null;
}

export interface FlashcardGenResult {
  cards: Flashcard[];
  rawContent: string;
  modelUsed: string;
}

// ── Card roles ────────────────────────────────────────────────────────────────
// "core"    → high-frequency sentence builders: I, you, want, need, like, more, no, yes, help, stop, go
// "content" → topic-specific nouns / verbs / adjectives: hungry, school, happy, tired …
// "phrase"  → short 2-word combos that are useful as a unit: "not happy", "want more", "all done"
type CardRole = 'core' | 'content' | 'phrase';
// Fitzgerald Key skin tone categories
type FitzCategory =
  | 'fitz_1' // Pale white
  | 'fitz_2' // Fair
  | 'fitz_3' // Medium
  | 'fitz_4' // Olive
  | 'fitz_5' // Brown
  | 'fitz_6' // Dark brown
  | null;

interface FlashcardGenOptions {
  relatedPrompts?: string[];
  previousAnswers?: string[];
  hardBan?: string[];        // never appear regardless of context (replaces driftBlock for strict bans)
  softDeprioritize?: string[]; // deprioritize but allow if highly relevant
  strictSymbols?: boolean;
  preferSymbols?: boolean;
}

// ── OpenAI client ─────────────────────────────────────────────────────────────
let openaiClient: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (openaiClient) return openaiClient;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Missing OPENAI_API_KEY');
  openaiClient = new OpenAI({ apiKey: key });
  return openaiClient;
}

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_CONTEXT_TOKENS = Number(process.env.MAX_CONTEXT_TOKENS || 450);

// Core vocabulary that should always be available for sentence building
// regardless of topic — these are the "glue" words of AAC communication
const CORE_VOCAB = new Set([
  'i', 'you', 'we', 'me', 'my', 'it', 'he', 'she', 'they',
  'want', 'need', 'like', 'love', 'feel', 'am', 'is', 'was',
  'have', 'do', 'go', 'help', 'stop', 'more', 'done', 'no',
  'yes', 'not', 'and', 'but', 'or', 'can', 'will', 'please',
]);

// These are useless as standalone AAC buttons
const ALWAYS_STRIP = new Set([
  'very', 'really', 'just', 'so', 'quite', 'kinda', 'kind',
  'of', 'a', 'an', 'the', 'that', 'this', 'those', 'these',
]);

// ── Helpers ───────────────────────────────────────────────────────────────────
function trimContext(raw: string): string {
  const parts = raw.split(/\s+/);
  return parts.length <= MAX_CONTEXT_TOKENS
    ? raw
    : parts.slice(0, MAX_CONTEXT_TOKENS).join(' ');
}

function normalise(s: string): string {
  return s
    .replace(/["""',.?!;:()\-]/g, ' ')
    .toLowerCase()
    .replace(/\bi['']m\b/g, 'im')
    .replace(/\bdon['']t\b/g, 'dont')
    .replace(/\bcan['']t\b/g, 'cant')
    .replace(/\bwon['']t\b/g, 'wont')
    .split(/\s+/)
    .filter(Boolean)
    .join(' ');
}

function dedupeLower(arr: string[]): string[] {
  const seen = new Set<string>();
  return arr.filter(a => {
    const k = a.toLowerCase().trim();
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function extractArray(raw: string): any[] {
  try {
    const p = JSON.parse(raw);
    if (Array.isArray(p)) return p;
    if (Array.isArray(p?.cards)) return p.cards;
  } catch {}
  const m = raw.match(/\[[\s\S]*\]/);
  if (m) {
    try {
      const p = JSON.parse(m[0]);
      if (Array.isArray(p)) return p;
    } catch {}
  }
  return [];
}

// ── cleanAnswer (the critical fix) ───────────────────────────────────────────
//
// OLD PROBLEMS:
//   1. slice(0,2) hard-clamp destroyed valid 2-word phrases
//   2. driftBlock nuked words entirely even when contextually relevant
//   3. strict regex /^[a-z ]{1,40}$/ blocked valid apostrophe words
//   4. No concept of card role — everything was judged by the same rules
//
// NEW APPROACH:
//   • Validate by role (core / phrase / content have different length rules)
//   • hardBan = absolute veto; softDeprioritize = lower priority but allowed
//   • Never strip a word just because it's in driftBlock if context contains it

function cleanAnswer(
  raw: string,
  role: CardRole,
  contextLC: string,
  hardBan: Set<string>,
  softDeprioritize: Set<string>,
): string | null {
  let a = normalise(raw);
  if (!a) return null;

  const words = a.split(' ').filter(Boolean);

  // Role-based length gates
  if (role === 'core' && words.length > 1) return null;       // core = single word only
  if (role === 'content' && words.length > 2) return null;    // content = 1–2 words
  if (role === 'phrase' && (words.length < 2 || words.length > 3)) return null; // phrase = 2–3 words

  // Strip always-useless filler words
  const filtered = words.filter(w => !ALWAYS_STRIP.has(w));
  if (filtered.length === 0) return null;
  a = filtered.join(' ');

  // Hard ban — absolute veto regardless of context
  for (const w of filtered) {
    if (hardBan.has(w)) return null;
  }

  // Soft deprioritize — allow if the word appears in context (contextually earned)
  // Returns null here so caller can push it to a lower-priority bucket
  const isDeprioritized = filtered.some(
    w => softDeprioritize.has(w) && !contextLC.includes(w)
  );
  if (isDeprioritized) return null; // caller can re-attempt with softer rules if needed

  // Final sanity: only plain text, allow apostrophes for contractions
  if (!/^[a-z' ]{1,40}$/.test(a)) return null;

  return a;
}

// ── Main export ───────────────────────────────────────────────────────────────
export async function generateFlashcards(
  context: string,
  requestedCount: number,
  maxCount: number,
  promptVersion: number,
  answerLength: string = 'mixed',
  recentAvg: number | null = null,
  options: FlashcardGenOptions = {},
  availableSymbols: string[] = [],
) {
  context = trimContext(context);
  const contextLC = context.toLowerCase();

  const hardBan = new Set((options.hardBan || []).map(w => normalise(w)));
  const softDeprioritize = new Set((options.softDeprioritize || []).map(w => normalise(w)));

  const previousAnswers = dedupeLower(
    (options.previousAnswers || [])
      .map(a => normalise(a))
      .filter(a => a.length > 0)
      .slice(0, 40)
  );

  const symbolList = availableSymbols
    .slice(0, 150)
    .map(s => s.replace('.svg', '').replace(/_/g, ' ').toLowerCase())
    .join(', ');

  // How many of each role to request
  // ~30% core (sentence glue), ~40% content (topic words), ~30% phrase (useful combos)
  const coreCount  = Math.max(2, Math.round(requestedCount * 0.30));
  const phraseCount = Math.max(2, Math.round(requestedCount * 0.25));
  const contentCount = requestedCount - coreCount - phraseCount;

  // ── Prompt ──────────────────────────────────────────────────────────────────
  //
  // KEY DESIGN DECISIONS:
  //  1. Explicit role system tells model EXACTLY what type of card to generate
  //     → Fixes the "all individual words" vs "all random phrases" inconsistency
  //  2. "Imagine 3 full sentences" chain-of-thought is preserved but now feeds
  //     into role-aware extraction instead of generic word dumping
  //  3. Ban system is explicit and tiered in the prompt itself
  //  4. Symbol list is a soft preference, not a hard constraint
  //
  const prompt = `
Each card should include:
  - question: string
  - answer: string
  - role: 'core' | 'content' | 'phrase'
  - fitz: 'fitz_1' | 'fitz_2' | 'fitz_3' | 'fitz_4' | 'fitz_5' | 'fitz_6' | null

Fitzgerald Key categories:
  fitz_1: Pale white
  fitz_2: Fair
  fitz_3: Medium
  fitz_4: Olive
  fitz_5: Brown
  fitz_6: Dark brown

You generate AAC (Augmentative and Alternative Communication) vocabulary flashcards.
A non-speaking user will tap these cards to build sentences and express themselves.

═══════════════════════════════════════════
STEP 1 — Understand the situation
═══════════════════════════════════════════
USER PROMPT: "${context}"

Imagine 3 things the user might want to say in response to this situation.
Write them as full sentences in your head (do NOT output them).
Example: if prompt is "how are you?" → think "I am good", "I feel tired", "I want water"

═══════════════════════════════════════════
STEP 2 — Generate 3 types of cards
═══════════════════════════════════════════
From those imagined sentences, extract words into these three roles:

ROLE: "core"  (generate exactly ${coreCount})
  • High-frequency sentence-building words the user needs to say ANYTHING
  • Examples: I, you, want, need, feel, am, not, more, help, stop, yes, no, done
  • Must be 1 word only
  • Always include relevant ones even if not in the symbol list

ROLE: "content"  (generate exactly ${contentCount})
  • Topic-specific words for THIS situation — nouns, verbs, adjectives
  • Examples for "how are you?": good, tired, hungry, happy, sad, sick
  • 1–2 words max
  • These change with every prompt — be specific to the situation

ROLE: "phrase"  (generate exactly ${phraseCount})
  • Short 2–3 word combos that are useful as a single tap
  • Examples: "not happy", "want more", "all done", "i want", "feel sick"
  • Only use 2–3 words, never single words in this role

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
✓ Every card must be something the user could actually SAY or TAP to communicate
✓ Plain English only — no punctuation, no emojis, no abbreviations (except: ok, dont, cant)
✓ No meta-words: never output "answer", "question", "topic", "sentence", "word", "phrase"
✗ HARD BAN — never output these regardless of context: ${Array.from(hardBan).join(', ') || 'none'}
⚠ AVOID unless clearly relevant to prompt: ${Array.from(softDeprioritize).join(', ') || 'none'}
${previousAnswers.length > 0 ? `↺ Already shown recently (avoid duplicating): ${previousAnswers.slice(0, 20).join(', ')}` : ''}
${symbolList ? `◈ Prefer these vocabulary symbols when relevant (but you MAY use other words): ${symbolList}` : ''}

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════
Return ONLY valid JSON, no markdown, no extra text:
{
  "cards": [
    {"question": "${context}", "answer": "I", "role": "core"},
    {"question": "${context}", "answer": "good", "role": "content"},
    {"question": "${context}", "answer": "feel tired", "role": "phrase"}
  ]
}
`.trim();

  const modelUsed = process.env.FAST_MODEL_NAME || 'gpt-4o-mini';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 0.15);
  const openai = getOpenAI();

  const completion = await openai.chat.completions.create({
    model: modelUsed,
    temperature,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content: `You are an AAC communication assistant. 
Output ONLY the JSON object specified. No preamble, no explanation, no markdown.
Every "answer" must be words a non-speaking person would tap to communicate — never meta-labels.
Respect the role rules strictly: core=1 word, content=1-2 words, phrase=2-3 words.`.trim()
      },
      { role: 'user', content: prompt }
    ]
  });

  const raw = completion.choices?.[0]?.message?.content || '{"cards":[]}';

  // ── Parse & filter ──────────────────────────────────────────────────────────
  let arr: any[] = [];
  try {
    const obj = JSON.parse(raw);
    arr = Array.isArray(obj.cards) ? obj.cards : extractArray(raw);
  } catch {
    arr = extractArray(raw);
  }

  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  // First pass: strict validation per role
  for (const o of arr) {
    if (!o || typeof o.answer !== 'string') continue;
    const role: CardRole = (['core','content','phrase'].includes(o.role)) ? o.role : 'content';
    const cleaned = cleanAnswer(o.answer, role, contextLC, hardBan, softDeprioritize);
    if (!cleaned) continue;
    if (seen.has(cleaned)) continue;
    seen.add(cleaned);
    cards.push({ question: context, answer: cleaned, role, fitz: o.fitz ?? null });
    if (cards.length >= requestedCount) break;
  }

  // Second pass: if still under-count, retry with softDeprioritize relaxed
  // (allow soft-banned words if they appeared in context)
  if (cards.length < requestedCount) {
    const emptySoft = new Set<string>(); // no soft bans this round
    for (const o of arr) {
      if (cards.length >= requestedCount) break;
      if (!o || typeof o.answer !== 'string') continue;
      const role: CardRole = (['core','content','phrase'].includes(o.role)) ? o.role : 'content';
      const cleaned = cleanAnswer(o.answer, role, contextLC, hardBan, emptySoft);
      if (!cleaned) continue;
      if (seen.has(cleaned)) continue;
      seen.add(cleaned);
      cards.push({ question: context, answer: cleaned, role, fitz: o.fitz ?? null });
    }
  }

  // Minimal symbol fallback: only if critically empty
  if (cards.length < Math.ceil(requestedCount * 0.2) && availableSymbols.length > 0) {
    for (const sym of availableSymbols.map(s => s.replace('.svg','').replace(/_/g,' ').toLowerCase())) {
      if (cards.length >= requestedCount) break;
      const n = normalise(sym);
      if (!n || seen.has(n) || hardBan.has(n)) continue;
      seen.add(n);
      cards.push({ question: context, answer: n, role: 'content', fitz: null });
    }
  }

  // Sort: core first, then content, then phrase — natural sentence-building order
  const roleOrder: Record<CardRole, number> = { core: 0, content: 1, phrase: 2 };
  cards.sort((a, b) => roleOrder[a.role ?? 'content'] - roleOrder[b.role ?? 'content']);

  if (cards.length > requestedCount) cards.length = requestedCount;

  console.log('[GEN STATS]', {
    requested: requestedCount,
    produced: cards.length,
    byRole: {
      core: cards.filter(c => c.role === 'core').length,
      content: cards.filter(c => c.role === 'content').length,
      phrase: cards.filter(c => c.role === 'phrase').length,
    },
    hardBanned: Array.from(hardBan).slice(0, 8),
    softDeprio: Array.from(softDeprioritize).slice(0, 8),
  });

  return { cards, rawContent: raw, modelUsed };
}