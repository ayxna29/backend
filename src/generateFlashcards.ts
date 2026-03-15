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
  if (words.length > 3) return null;
  const filtered = words.filter(w => !ALWAYS_STRIP.has(w));
  if (filtered.length === 0) return null;
  a = filtered.join(' ');
  if (hardBan.has(a)) return null;
  if (!/^[a-z' ]{1,40}$/.test(a)) return null;
  return a;
}

// Non-streaming version (kept for internal use / fallback)
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
  const { cards, rawContent, modelUsed } = await generateFlashcardsStream(
    context, requestedCount, maxCount, promptVersion, answerLength,
    recentAvg, options, availableSymbols,
    () => {} // no-op card callback
  );
  return { cards, rawContent, modelUsed };
}

// Streaming version — calls onCard(card) each time a new card is ready
export async function generateFlashcardsStream(
  context: string,
  requestedCount: number,
  maxCount: number,
  promptVersion: number,
  answerLength: string = 'mixed',
  recentAvg: number | null = null,
  options: FlashcardGenOptions = {},
  availableSymbols: string[] = [],
  onCard: (card: Flashcard) => void,
): Promise<FlashcardGenResult> {
  context = trimContext(context);
  const hardBan = new Set((options.hardBan || []).map(w => normalise(w)));
  const previousAnswers = dedupeLower(
    (options.previousAnswers || []).map(a => normalise(a)).filter(a => a.length > 0).slice(0, 40)
  );

  const prompt = [
    `You generate AAC (Augmentative and Alternative Communication) vocabulary flashcards.`,
    `A non-speaking person taps these cards one at a time to build sentences and communicate.`,
    ``,
    `SITUATION: "${context}"`,
    ``,
    `Your task is to generate vocabulary that would realistically be used by someone responding in this situation.`,
    ``,
    `The cards should work together so the user can tap multiple cards in sequence to form responses.`,
    `Think about the words someone might use to answer, react, explain, or participate in this situation.`,
    ``,
    `Every card must clearly relate to the situation.`,
    `Avoid random words that would not make sense in a response.`,
    ``,
    `Examples`,
    ``,
    `Situation: "how are you feeling"`,
    `Possible vocabulary: i, am, feel, bad, good, amazing, better, today`,
    ``,
    `Situation: "what is your name"`,
    `Possible vocabulary: i, am, my, name, called, dont know`,
    ``,
    `Guidelines`,
    `- Cards should combine naturally with other cards to form responses.`,
    `- Prefer single words but short 2-word phrases are allowed if commonly used together.`,
    `- All answers must be lowercase.`,
    `- No duplicates.`,
    ``,
    `FITZGERALD KEY — assign a category to each card`,
    `  "person"     -> i, you, he, she, we, they, me, my`,
    `  "verb"       -> am, is, feel, want, need, go, eat, help, do, have`,
    `  "descriptor" -> good, bad, happy, sad, tired, sick, better, okay`,
    `  "noun"       -> people, places, or things relevant to the situation`,
    `  "social"     -> yes, no, please, thank you, more, stop, done`,
    `  "question"   -> what, where, when, why, who, how`,
    ``,
    `Return exactly ${requestedCount} cards.`,
    ``,
    `OUTPUT JSON ONLY`,
    `{"cards":[{"question":"${context}","answer":"word","fitz":"person|verb|descriptor|noun|social|question"}]}`
  ].join('\n');

  const modelUsed = process.env.FAST_MODEL_NAME || process.env.MODEL_NAME || 'gpt-5-nano';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 1);
  const openai = getOpenAI();

  // Use streaming completion
  const stream = await openai.chat.completions.create({
    model: modelUsed,
    temperature,
    stream: true,
    messages: [
      {
        role: 'system',
        content: [
          'You are an AAC communication assistant generating situation-specific vocabulary.',
          'Output ONLY the JSON object specified. No preamble, no explanation, no markdown.',
          'CRITICAL: Every single card must be a word this person would actually use to respond to the exact situation given.',
          'Think: if someone asked me this question, what words would I tap to answer it? EVERY FLASHCARD/WORD MUST BE JUSTIFIABLE AS A REAL RESPONSE TO THE SITUATION.',
        ].join('\n'),
      },
      { role: 'user', content: prompt }
    ]
  });

  const VALID_FITZ = new Set(['person', 'verb', 'descriptor', 'noun', 'social', 'question']);
  const fitzCounts = new Map<string, number>();
  // Raised cap: allow up to 60% per category so one dominant type doesn't
  // cause everything to be thrown away and trigger a 422
  const MAX_PER_FITZ = Math.max(8, Math.ceil(requestedCount * 0.60));
  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  let rawContent = '';
  let buffer = '';

  // Parse cards incrementally as JSON chunks stream in.
  // Strategy: look for complete {"question":...,"answer":...,"fitz":...} objects in the buffer.
  const cardRegex = /\{[^{}]*?"answer"\s*:\s*"([^"]+)"[^{}]*?"fitz"\s*:\s*"([^"]+)"[^{}]*?\}/g;
  const cardRegex2 = /\{[^{}]*?"fitz"\s*:\s*"([^"]+)"[^{}]*?"answer"\s*:\s*"([^"]+)"[^{}]*?\}/g;

  function tryExtractNewCards() {
    // Try both field orderings
    for (const regex of [cardRegex, cardRegex2]) {
      regex.lastIndex = 0;
      let match;
      while ((match = regex.exec(buffer)) !== null) {
        const answerRaw = regex === cardRegex ? match[1] : match[2];
        const fitzRaw   = regex === cardRegex ? match[2] : match[1];

        const cleaned = cleanAnswer(answerRaw, hardBan);
        if (!cleaned || seen.has(cleaned)) continue;
        if (!VALID_FITZ.has(fitzRaw)) continue;
        const currentCount = fitzCounts.get(fitzRaw) || 0;
        if (currentCount >= MAX_PER_FITZ) continue;

        seen.add(cleaned);
        fitzCounts.set(fitzRaw, currentCount + 1);
        const card: Flashcard = { question: context, answer: cleaned, fitz: fitzRaw };
        cards.push(card);
        onCard(card); // 🔥 emit card immediately
        if (cards.length >= requestedCount) return;
      }
    }
  }

  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content || '';
    rawContent += delta;
    buffer += delta;

    // Only try to parse once we have a reasonable chunk (avoid thrashing)
    if (buffer.includes('"answer"') && buffer.includes('"fitz"')) {
      tryExtractNewCards();
      if (cards.length >= requestedCount) break;
    }
  }

  // Final pass on complete buffer — catch anything missed during streaming
  if (cards.length < requestedCount) {
    buffer = rawContent;
    tryExtractNewCards();
  }

  // Second pass — relax diversity cap
  if (cards.length < requestedCount) {
    const arr = extractArray(rawContent);
    for (const o of arr) {
      if (cards.length >= requestedCount) break;
      if (!o || typeof o.answer !== 'string') continue;
      const cleaned = cleanAnswer(o.answer, hardBan);
      if (!cleaned || seen.has(cleaned)) continue;
      const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : 'noun';
      seen.add(cleaned);
      const card: Flashcard = { question: context, answer: cleaned, fitz };
      cards.push(card);
      onCard(card);
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

  return { cards, rawContent, modelUsed };
}