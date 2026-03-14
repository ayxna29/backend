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

// Never strip i, am, my, me — critical AAC vocabulary
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
    .replace(/["""',.?!;()\-]/g, ' ')
    .toLowerCase()
    .replace(/\bi['\u2019]m\b/g, 'im')
    .replace(/\bdon['\u2019]t\b/g, 'dont')
    .replace(/\bcan['\u2019]t\b/g, 'cant')
    .replace(/\bwon['\u2019]t\b/g, 'wont')
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
    ? `- NEVER include: ${Array.from(hardBan).join(', ')}`
    : '';
  const prevLine = previousAnswers.length > 0
    ? `- Already shown recently, skip: ${previousAnswers.slice(0, 15).join(', ')}`
    : '';

  // Situation classifier
  const ctx = context.toLowerCase();
  const isGreeting = /how are you|how r you|how are u|how do you feel|how was your|how did you|good morning|good night|good afternoon|how.s it going|how have you|what.s up|feeling today/.test(ctx);
  const isFood = /eat|eating|ate|eaten|hungry|hunger|food|breakfast|lunch|dinner|supper|snack|drink|drinking|drank|thirsty|meal|cook|bake|baking|pizza|burger|sandwich|cereal|restaurant|cafe|kitchen|taste|flavor|yummy|delicious|apple|pasta|rice|soup|fruit|veggie/.test(ctx);
  const isPain = /hurt|hurting|pain|painful|sore|sick|ill|unwell|doctor|hospital|nurse|medicine|ache|fever|ouch|injury|injure|dizzy|nausea|nauseous|vomit|headache|stomachache|not feel|don.t feel|feel bad|feeling bad|feel sick/.test(ctx);
  const isActivity = /play|playing|game|games|watch|watching|tv|movie|film|go|going|outside|park|school|class|lesson|sport|swim|swimming|run|running|bike|biking|read|reading|book|draw|drawing|music|dance|dancing|craft|build/.test(ctx);
  const isNeed = /want|need|help|bathroom|toilet|potty|water|tired|sleep|sleeping|rest|resting|stop|done|finish|more|again|break|wait|ready|not ready/.test(ctx);
  const isEmotion = /feel|feeling|emotion|happy|sad|angry|mad|scared|fear|worried|worry|excited|upset|calm|bored|lonely|nervous|anxious|proud|love|hate|frustrated|overwhelm/.test(ctx);

  let situationGuide: string;
  if (isGreeting) {
    situationGuide = [
      'SITUATION TYPE: Greeting / check-in',
      'WHAT BELONGS HERE:',
      '- Feeling words — include BOTH positive and negative emotions. Do not lean negative.',
      '  Positive: excited, great, happy, good, loved, proud, calm, wonderful, silly, amazing',
      '  Negative: tired, sad, sick, scared, nervous, bored, confused, upset',
      '  Pick a realistic mix',
      '- Verbs: am, feel, doing, having, need',
      '- Social: yes, no, thank you',
      '- Person connectors: i, my',
      'WHAT DOES NOT BELONG: body parts, food, school, home, friend, random nouns',
    ].join('\n');
  } else if (isFood) {
    situationGuide = [
      'SITUATION TYPE: Food / eating',
      'WHAT BELONGS HERE:',
      '- Food/drink words: start with anything mentioned, then freely add varied foods — pizza, pasta, juice, apple, sandwich, cereal, soup, water, snack, fruit, bread. Be creative.',
      '- Feeling words: hungry, full, yummy, yuck, like, love, want',
      '- Verbs: want, eat, drink, have, like, need, make, taste',
      '- Social: please, more, done, stop, help',
      'WHAT DOES NOT BELONG: unrelated emotions, body parts, places',
    ].join('\n');
  } else if (isPain) {
    situationGuide = [
      'SITUATION TYPE: Pain / feeling unwell',
      'WHAT BELONGS HERE:',
      '- Body parts: include parts mentioned or implied. For general unwell situations, common parts like head, stomach, throat, arm, leg are fair since person may need to point to where it hurts.',
      '- Pain descriptors: hurt, sore, bad, sick, tired, scared, better, worse, okay, fine, awful, dizzy',
      '- Verbs: hurt, feel, need, want, help, stop, rest',
      '- Social: please, help, yes, no',
      '- Include positive options too: better, okay, fine',
      'WHAT DOES NOT BELONG: food, school, unrelated places',
    ].join('\n');
  } else if (isActivity) {
    situationGuide = [
      'SITUATION TYPE: Activity / play / going somewhere',
      'WHAT BELONGS HERE:',
      '- Activity/place words specific to the situation',
      '- Verbs: go, play, want, like, watch, do, can, come',
      '- Person connectors: i, my, you, we',
      '- Social: please, yes, no, more, done',
      'WHAT DOES NOT BELONG: food, body parts, emotions unrelated to the activity',
    ].join('\n');
  } else if (isNeed) {
    situationGuide = [
      'SITUATION TYPE: Expressing a need',
      'WHAT BELONGS HERE:',
      '- Need words from the context',
      '- Verbs: want, need, help, stop, go, do',
      '- Person connectors: i, my, you',
      '- Social: please, yes, no, more, done, help',
      'WHAT DOES NOT BELONG: unrelated nouns, random emotions, body parts',
    ].join('\n');
  } else {
    situationGuide = [
      `SITUATION TYPE: General — responding to: "${context}"`,
      'Before adding any word, ask: would this person actually tap this to respond?',
      'If the answer is probably not — leave it out. Every word must earn its place.',
    ].join('\n');
  }

  const prompt = [
    'You generate AAC (Augmentative and Alternative Communication) vocabulary flashcards.',
    'A non-speaking person taps these cards one at a time to build sentences and communicate.',
    '',
    `SITUATION: "${context}"`,
    '',
    situationGuide,
    '',
    'STEP 1 - Internally write 5 complete sentences this person might tap out.',
    `Example for "how are you today": "i am good", "i feel tired", "i am excited", "i need help", "i feel happy"`,
    'Include positive emotions, not just negative. Make sentences varied.',
    '',
    'STEP 2 - Extract every word needed to build those sentences as individual cards.',
    'For each sentence, check every word is covered. The user must be able to reconstruct those sentences card by card.',
    '',
    `SECTION A — CONTENT WORDS (exactly ${contentCount} cards)`,
    'The meaningful words: feelings, actions, objects, descriptors for this situation.',
    'Follow the SITUATION TYPE guide above.',
    'Include a RANGE of emotions — both positive (excited, great, proud, loved) and negative (sad, tired, scared).',
    '',
    `SECTION B — SENTENCE CONNECTORS (exactly ${coreCount} cards)`,
    'Words that glue sentences together: pronouns, linking verbs, function words.',
    'Extract from the sentences in step 1. Always include: i, am (or feel/is as needed).',
    '',
    'MULTI-WORD: fixed phrases allowed: "ice cream", "all done", "thank you", "good morning"',
    '',
    'RULES:',
    '- Answers: 1 word or 1 fixed phrase (max 2 words), lowercase',
    '- No punctuation except spaces in 2-word phrases',
    '- No duplicates',
    '- No meta-words: answer, question, sentence, example, word, topic',
    hardBanLine,
    prevLine,
    '',
    'FITZGERALD KEY — add "fitz" to every card:',
    '  "person"     -> i, you, he, she, we, they, me, my, mom, dad',
    '  "verb"       -> am, is, feel, want, need, go, eat, help, do, have, hurt, like',
    '  "descriptor" -> happy, excited, great, proud, calm, loved, silly, nervous, confused, bored, surprised, grateful, sad, angry, scared, tired, sick, okay, fine, better, hot, cold',
    '  "noun"       -> things/objects/places specific to the situation',
    '  "social"     -> yes, no, please, more, stop, done, again, wait, sorry, thank you',
    '  "question"   -> what, where, when, why, who, how',
    '',
    'OUTPUT — return ONLY valid JSON, no markdown:',
    '{"cards": [{"question": "...", "answer": "...", "fitz": "..."}]}',
    '',
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
          'You are an AAC communication assistant generating situation-specific vocabulary.',
          'Output ONLY the JSON object specified. No preamble, no explanation, no markdown.',
          'CRITICAL: Every card must be directly relevant to the specific situation provided.',
          'Follow the SITUATION TYPE guide in the prompt — it tells you exactly what categories to use.',
          'Do NOT fall back to generic vocabulary (random body parts, food, school) unless the situation demands it.',
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
    const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : 'noun';
    const currentCount = fitzCounts.get(fitz) || 0;
    if (currentCount >= MAX_PER_FITZ) continue;
    seen.add(cleaned);
    fitzCounts.set(fitz, currentCount + 1);
    cards.push({ question: context, answer: cleaned, fitz });
    if (cards.length >= requestedCount) break;
  }

  if (cards.length < requestedCount) {
    for (const o of arr) {
      if (cards.length >= requestedCount) break;
      if (!o || typeof o.answer !== 'string') continue;
      const cleaned = cleanAnswer(o.answer, hardBan);
      if (!cleaned || seen.has(cleaned)) continue;
      seen.add(cleaned);
      const fitz = VALID_FITZ.has(o.fitz) ? o.fitz as string : 'noun';
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