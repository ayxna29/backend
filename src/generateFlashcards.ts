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
  'of', 'a', 'an', 'the', 'that', 'this', 'those', 'these',
]);

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

function cleanAnswer(raw: string, hardBan: Set<string>): string | null {
  let a = normalise(raw);
  if (!a) return null;

  const words = a.split(' ').filter(Boolean);

  // Single words only
  if (words.length > 1) return null;

  const filtered = words.filter(w => !ALWAYS_STRIP.has(w));
  if (filtered.length === 0) return null;
  a = filtered[0];

  if (hardBan.has(a)) return null;

  if (!/^[a-z']{1,40}$/.test(a)) return null;

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
    (options.previousAnswers || [])
      .map(a => normalise(a))
      .filter(a => a.length > 0)
      .slice(0, 40)
  );

  // 70% context words, 30% sentence builders
  // For 30 cards: ~21 topic words covering the situation broadly + ~9 glue words
  const contentCount = Math.max(2, Math.round(requestedCount * 0.70));
  const coreCount = requestedCount - contentCount;

  const hardBanLine = Array.from(hardBan).length > 0
    ? `- NEVER include: ${Array.from(hardBan).join(', ')}`
    : '';
  const prevLine = previousAnswers.length > 0
    ? `- Already shown recently, skip: ${previousAnswers.slice(0, 15).join(', ')}`
    : '';

  const prompt = [
    `You generate AAC (Augmentative and Alternative Communication) vocabulary flashcards.`,
    `A non-speaking person taps these cards one at a time to build sentences and communicate.`,
    ``,
    `SITUATION: "${context}"`,
    ``,
    `STEP 1 - Imagine 6 different things this person might want to say. Do NOT output them.`,
    `Think broadly - different needs, emotions, and responses all related to the situation.`,
    ``,
    `STEP 2 - Extract SINGLE WORDS only. Never put 2 or 3 words on one card.`,
    ``,
    `SECTION A - CONTEXT WORDS (generate exactly ${contentCount})`,
    `Single words directly relevant to this situation. Cover the full range of what someone might want to express.`,
    `Order them: most immediately useful and specific first, broader/secondary words later.`,
    `Use ALL ${contentCount} slots - go deep into the topic, don't repeat, don't pad with generic words.`,
    ``,
    `SECTION B - SENTENCE BUILDERS (generate exactly ${coreCount})`,
    `From the 6 sentences you imagined, extract the glue words that hold them together.`,
    `Infer what's needed - pronouns, verbs, linking words. Every situation needs different builders.`,
    ``,
    `RULES:`,
    `- Every answer = exactly 1 word, lowercase, no punctuation`,
    `- Never output meta-words: answer, question, topic, sentence, response, example, word`,
    `- No duplicates`,
    hardBanLine,
    prevLine,
    ``,
    `FITZGERALD KEY - add "fitz" to every card:`,
    `  "person"     -> i, you, he, she, we, they, me, my, mom, dad, friend`,
    `  "verb"       -> actions/states: hurt, feel, am, is, want, need, go, eat, help, do, have`,
    `  "descriptor" -> adjectives/feelings: good, bad, happy, sad, tired, sick, sore, better, hot, cold`,
    `  "noun"       -> things/places/body parts: head, home, food, school, arm, stomach, water`,
    `  "social"     -> yes, no, please, more, stop, done, help, again, wait, sorry`,
    `  "question"   -> what, where, when, why, who, how`,
    ``,
    `OUTPUT - return ONLY valid JSON, no markdown:`,
    `{`,
    `  "cards": [`,
    `    {"question": "${context}", "answer": "head", "fitz": "noun"},`,
    `    {"question": "${context}", "answer": "stomach", "fitz": "noun"},`,
    `    {"question": "${context}", "answer": "hurts", "fitz": "verb"},`,
    `    {"question": "${context}", "answer": "i", "fitz": "person"},`,
    `    {"question": "${context}", "answer": "my", "fitz": "person"}`,
    `  ]`,
    `}`,
    ``,
    `Generate exactly ${requestedCount} cards. Section A first (context words), Section B last (sentence builders).`,
  ].filter(l => l !== null && l !== undefined).join('\n');

  const modelUsed = process.env.FAST_MODEL_NAME || process.env.MODEL_NAME || 'gpt-4o-mini';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 0.15);
  const openai = getOpenAI();

  const completion = await openai.chat.completions.create({
    model: modelUsed,
    temperature,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content: [
          'You are an AAC communication assistant.',
          'Output ONLY the JSON object specified. No preamble, no explanation, no markdown.',
          'Every "answer" must be a single word a non-speaking person would tap to communicate.',
          'NEVER combine multiple words into one answer. Each card = exactly one word.',
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

  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  for (const o of arr) {
    if (!o || typeof o.answer !== 'string') continue;
    const cleaned = cleanAnswer(o.answer, hardBan);
    if (!cleaned) continue;
    if (seen.has(cleaned)) continue;
    seen.add(cleaned);
    const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : null;
    cards.push({ question: context, answer: cleaned, fitz });
    if (cards.length >= requestedCount) break;
  }

  // Second pass if under count
  if (cards.length < requestedCount) {
    for (const o of arr) {
      if (cards.length >= requestedCount) break;
      if (!o || typeof o.answer !== 'string') continue;
      const cleaned = cleanAnswer(o.answer, hardBan);
      if (!cleaned || seen.has(cleaned)) continue;
      seen.add(cleaned);
      const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : null;
      cards.push({ question: context, answer: cleaned, fitz });
    }
  }

  if (cards.length > requestedCount) cards.length = requestedCount;

  console.log('[GEN STATS]', {
    requested: requestedCount,
    produced: cards.length,
    model: modelUsed,
    sample: cards.slice(0, 6).map(c => `${c.answer}(${c.fitz})`),
  });

  return { cards, rawContent: raw, modelUsed };
}