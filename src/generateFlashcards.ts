import OpenAI from 'openai';

export interface Flashcard {
  question: string;
  answer: string;
  fitz?: string | null;
}

export interface FlashcardGenResult {
  cards: Flashcard[];
  rawContent: string;
  modelUsed: string;
}

interface FlashcardGenOptions {
  relatedPrompts?: string[];
  previousAnswers?: string[];
  hardBan?: string[];
  softDeprioritize?: string[];
  strictSymbols?: boolean;
  preferSymbols?: boolean;
}

let openaiClient: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (openaiClient) return openaiClient;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Missing OPENAI_API_KEY');
  openaiClient = new OpenAI({ apiKey: key });
  return openaiClient;
}

const MAX_CONTEXT_TOKENS = Number(process.env.MAX_CONTEXT_TOKENS || 450);

const ALWAYS_STRIP = new Set([
  'very', 'really', 'just', 'so', 'quite', 'kinda', 'kind',
  'of', 'an', 'the', 'that', 'this', 'those', 'these',
]);

function trimContext(raw: string): string {
  const parts = raw.split(/\s+/);
  return parts.length <= MAX_CONTEXT_TOKENS ? raw : parts.slice(0, MAX_CONTEXT_TOKENS).join(' ');
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

function cleanAnswer(raw: string, hardBan: Set<string>): string | null {
  let a = normalise(raw);
  if (!a) return null;
  const words = a.split(' ').filter(Boolean);
  if (words.length > 2) return null;
  const filtered = words.filter(w => !ALWAYS_STRIP.has(w));
  if (filtered.length === 0) return null;
  a = filtered.join(' ');
  if (hardBan.has(a)) return null;
  if (!/^[a-z' ]{1,40}$/.test(a)) return null;
  return a;
}

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
  const hardBan = new Set((options.hardBan || []).map(w => normalise(w)));
  const previousAnswers = dedupeLower(
    (options.previousAnswers || []).map(a => normalise(a)).filter(a => a.length > 0).slice(0, 40)
  );

  const contentCount = Math.max(2, Math.round(requestedCount * 0.65));
  const coreCount = requestedCount - contentCount;

  const hardBanLine = Array.from(hardBan).length > 0
    ? `- NEVER include: ${Array.from(hardBan).join(', ')}` : '';
  const prevLine = previousAnswers.length > 0
    ? `- Already shown recently, skip: ${previousAnswers.slice(0, 15).join(', ')}` : '';

  const prompt = [
    `You generate AAC (Augmentative and Alternative Communication) vocabulary flashcards.`,
    `A non-speaking person taps these cards one at a time to build sentences and communicate.`,
    ``,
    `SITUATION: "${context}"`,
    ``,
    `STEP 1 — Internally write 5 complete sentences this person would actually say in THIS situation.`,
    `Each sentence must be a direct response to the exact situation described.`,
    `Example for "what is your name": "my name is", "i am called", "my name", "what is your name", "i dont know"`,
    `Example for "how are you doing": "i am good", "i feel tired", "i am happy today", "i need help", "i feel okay"`,
    `Do NOT output these sentences.`,
    ``,
    `STEP 2 — Extract EVERY word needed to tap out those 5 sentences.`,
    `Every card must come directly from step 1. If a word wasn't in any sentence, don't include it.`,
    ``,
    `SECTION A — CONTENT WORDS (exactly ${contentCount} cards)`,
    `The meaningful words from your sentences: names, feelings, descriptors, objects SPECIFIC to this situation.`,
    `Ask yourself: "Would someone actually say this word when responding to THIS situation?"`,
    `If the answer is no or maybe — leave it out.`,
    ``,
    `SECTION B — SENTENCE CONNECTORS (exactly ${coreCount} cards)`,
    `The grammar glue words from your sentences: pronouns, linking verbs, function words.`,
    `Only include what appeared in your step 1 sentences. Always include: i, am (or feel/is as appropriate).`,
    ``,
    `MULTI-WORD: fixed phrases allowed as one card: "thank you", "all done", "good morning", "my name"`,
    ``,
    `STRICT RULES:`,
    `- Every single card must directly serve responding to: "${context}"`,
    `- No generic padding words that could appear in ANY AAC board (school, home, food, water, friend, play area — unless the situation specifically involves them)`,
    `- 1 word or 1 fixed phrase max 2 words, lowercase`,
    `- No duplicates`,
    hardBanLine,
    prevLine,
    ``,
    `FITZGERALD KEY — add "fitz" to every card:`,
    `  "person"     -> i, you, he, she, we, they, me, my, mom, dad, name`,
    `  "verb"       -> am, is, feel, want, need, go, eat, help, do, have, hurt, called, know`,
    `  "descriptor" -> good, bad, happy, sad, tired, sick, sore, better, okay, fine, excited, great, scared`,
    `  "noun"       -> things/places/objects SPECIFIC to the situation`,
    `  "social"     -> yes, no, please, more, stop, done, again, wait, sorry, thank you, hello`,
    `  "question"   -> what, where, when, why, who, how`,
    ``,
    `OUTPUT — return ONLY valid JSON, no markdown:`,
    `{"cards": [{"question": "...", "answer": "...", "fitz": "..."}]}`,
    ``,
    `Generate exactly ${requestedCount} cards. Section A first, Section B last.`,
  ].filter(l => l !== null && l !== undefined).join('\n');

  const modelUsed = process.env.FAST_MODEL_NAME || process.env.MODEL_NAME || 'gpt-5-nano';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 0.3);
  const openai = getOpenAI();

  const completion = await openai.chat.completions.create({
    model: modelUsed,
    temperature,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content: [
          'You are an AAC communication assistant generating situation-specific vocabulary.',
          'Output ONLY the JSON object specified. No preamble, no explanation, no markdown.',
          'CRITICAL: Every single card must be a word this person would actually use to respond to the exact situation given.',
          'Generic AAC words like school, home, food, water, friend are FORBIDDEN unless the situation explicitly involves them.',
          'Think: if someone asked me this question, what words would I tap to answer it? EVERY FLASHCARD/WORD MUST BE JUSTIFIABLE AS A REAL RESPONSE TO THE SITUATION.',
        ].join('\n'),
      },
      { role: 'user', content: prompt }
    ]
  });

  const raw = completion.choices?.[0]?.message?.content || '{"cards":[]}';

  let arr: any[] = [];
  try {
    const obj = JSON.parse(raw);
    arr = Array.isArray(obj.cards) ? obj.cards : extractArray(raw);
  } catch {
    arr = extractArray(raw);
  }

  const VALID_FITZ = new Set(['person', 'verb', 'descriptor', 'noun', 'social', 'question']);
  const fitzCounts = new Map<string, number>();
  const MAX_PER_FITZ = Math.max(4, Math.ceil(requestedCount * 0.35));

  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  for (const o of arr) {
    if (!o || typeof o.answer !== 'string') continue;
    const cleaned = cleanAnswer(o.answer, hardBan);
    if (!cleaned) continue;
    if (seen.has(cleaned)) continue;
    const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : null;
    if (!fitz) continue;
    const currentCount = fitzCounts.get(fitz) || 0;
    if (currentCount >= MAX_PER_FITZ) continue;
    seen.add(cleaned);
    fitzCounts.set(fitz, currentCount + 1);
    cards.push({ question: context, answer: cleaned, fitz });
    if (cards.length >= requestedCount) break;
  }

  // Second pass — relax diversity cap
  if (cards.length < requestedCount) {
    for (const o of arr) {
      if (cards.length >= requestedCount) break;
      if (!o || typeof o.answer !== 'string') continue;
      const cleaned = cleanAnswer(o.answer, hardBan);
      if (!cleaned || seen.has(cleaned)) continue;
      const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : 'noun';
      seen.add(cleaned);
      cards.push({ question: context, answer: cleaned, fitz });
    }
  }

  if (cards.length > requestedCount) cards.length = requestedCount;

  console.log('[GEN STATS]', {
    requested: requestedCount,
    produced: cards.length,
    model: modelUsed,
    fitzDist: Object.fromEntries(fitzCounts),
    sample: cards.slice(0, 6).map(c => `${c.answer}(${c.fitz})`),
  });

  return { cards, rawContent: raw, modelUsed };
}