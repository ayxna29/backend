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

// 'a' removed — keep all meaningful single-letter words like 'i'
const ALWAYS_STRIP = new Set([
  'very', 'really', 'just', 'so', 'quite', 'kinda', 'kind',
  'of', 'an', 'the', 'that', 'this', 'those', 'these',
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

  // Allow up to 2 words (for phrases like "ice cream", "all done", "thank you")
  if (words.length > 2) return null;

  const filtered = words.filter(w => !ALWAYS_STRIP.has(w));
  if (filtered.length === 0) return null;

  // Rejoin (handles both 1 and 2 word answers)
  a = filtered.join(' ');

  if (hardBan.has(a)) return null;

  // Allow letters, apostrophes, spaces (for 2-word phrases)
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
    (options.previousAnswers || [])
      .map(a => normalise(a))
      .filter(a => a.length > 0)
      .slice(0, 40)
  );

  // 65% context words, 35% sentence builders
  const contentCount = Math.max(2, Math.round(requestedCount * 0.65));
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
    `STEP 1 - Think of 6 DIFFERENT things this person might want to say in this situation.`,
    `Make them varied — different emotions, needs, responses, not all the same type.`,
    `Do NOT output these sentences.`,
    ``,
    `STEP 2 - Generate the cards below.`,
    ``,
    `SECTION A — CONTEXT WORDS (exactly ${contentCount} cards)`,
    `Words directly relevant to this specific situation.`,
    `CRITICAL DIVERSITY RULE: spread across multiple categories. Do NOT generate more than`,
    `2-3 words from the same Fitzgerald category. For example if the situation is about feelings,`,
    `pick 2-3 feeling descriptors max, then move on to relevant nouns, verbs, social words etc.`,
    `Think: what variety of words would help someone respond fully to this situation?`,
    ``,
    `SECTION B — SENTENCE BUILDERS (exactly ${coreCount} cards)`,
    `Core grammar words needed to form sentences about this situation.`,
    `Pick the ones most likely needed — don't just dump all pronouns.`,
    `Infer what's actually needed from the 6 sentences you imagined.`,
    ``,
    `MULTI-WORD EXCEPTION: common fixed phrases are allowed as a single card:`,
    `"ice cream", "all done", "thank you", "good morning", "play area"`,
    `These must be natural compound concepts, not random word combos.`,
    ``,
    `RULES:`,
    `- Answers: 1 word (or 1 fixed compound phrase max 2 words), lowercase`,
    `- No punctuation except spaces in 2-word phrases`,
    `- No duplicates`,
    `- No meta-words: answer, question, topic, sentence, response, example, word`,
    `- No irrelevant generic padding — every word must earn its place`,
    hardBanLine,
    prevLine,
    ``,
    `FITZGERALD KEY — add "fitz" field to every card:`,
    `  "person"     -> i, you, he, she, we, they, me, my, mom, dad, friend`,
    `  "verb"       -> actions/states: am, is, feel, want, need, go, eat, help, do, have, hurt`,
    `  "descriptor" -> adjectives/feelings: good, bad, happy, sad, tired, sick, sore, better, hot, cold`,
    `  "noun"       -> things/places/body parts: home, food, school, water, head, stomach`,
    `  "social"     -> yes, no, please, more, stop, done, help, again, wait, sorry, thank you`,
    `  "question"   -> what, where, when, why, who, how`,
    ``,
    `OUTPUT — return ONLY valid JSON, no markdown, no explanation:`,
    `{"cards": [{"question": "...", "answer": "...", "fitz": "..."}]}`,
    ``,
    `Generate exactly ${requestedCount} cards. Section A first, Section B last.`,
  ].filter(l => l !== null && l !== undefined).join('\n');

  const modelUsed = process.env.FAST_MODEL_NAME || process.env.MODEL_NAME || 'gpt-4o-mini';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 0.25);
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
          'Every "answer" must be a word or short fixed phrase a non-speaking person would tap.',
          'DIVERSITY IS CRITICAL',
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
  const MAX_PER_FITZ = Math.max(3, Math.ceil(requestedCount * 0.25));

  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  for (const o of arr) {
    if (!o || typeof o.answer !== 'string') continue;
    const cleaned = cleanAnswer(o.answer, hardBan);
    if (!cleaned) continue;
    if (seen.has(cleaned)) continue;

    // Use AI's fitz value — don't default everything to noun
    const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : null;
    if (!fitz) continue; // skip cards with no valid fitz — AI must provide it

    const currentCount = fitzCounts.get(fitz) || 0;
    if (currentCount >= MAX_PER_FITZ) continue;

    seen.add(cleaned);
    fitzCounts.set(fitz, currentCount + 1);
    cards.push({ question: context, answer: cleaned, fitz });
    if (cards.length >= requestedCount) break;
  }

  // Second pass — relax diversity, still use AI's fitz not a default
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