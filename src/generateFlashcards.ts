import OpenAI from 'openai';

export interface Flashcard { question: string; answer: string; }
export interface FlashcardGenResult {
  cards: Flashcard[];
  rawContent: string;
  modelUsed: string;
}

interface FlashcardGenOptions {
  relatedPrompts?: string[];
  previousAnswers?: string[];
  driftBlock?: string[];
  strictSymbols?: boolean; 
  preferSymbols?: boolean; // prefer symbol vocabulary but don't strictly enforce
}

let openaiClient: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (openaiClient) return openaiClient;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Missing OPENAI_API_KEY');
  openaiClient = new OpenAI({ apiKey: key });
  return openaiClient;
}

// ------------------- Helpers -------------------
const MAX_CONTEXT_TOKENS = Number(process.env.MAX_CONTEXT_TOKENS || 450);

function trimContext(raw: string): string {
  const parts = raw.split(/\s+/);
  if (parts.length <= MAX_CONTEXT_TOKENS) return raw;
  return parts.slice(0, MAX_CONTEXT_TOKENS).join(' ');
}

function clampShortRaw(ans: string): string {
  return ans
    .replace(/["""',.?!;:]/g, ' ')
    .toLowerCase()
    .replace(/\bi['']m\b/g, 'im')
    .split(/\s+/)
    .filter(Boolean)
    .join(' ');
}

const FILLER = new Set(['very','some','really','just','so','quite','kinda','kind','of']);

function cleanAnswer(ans: string, contextLC: string, driftBlock: Set<string>, strict = true): string | null {
  let a = ans.trim();
  if (!a) return null;

  if (!strict) return a;

  a = clampShortRaw(a);
  if (!a) return null;

  // Allow up to 3 words but clamp to first 2
  const words = a.split(' ').filter(Boolean);
  if (words.length > 3) return null;

  let filtered = words.filter(w => !FILLER.has(w));
  if (filtered.length === 0) return null;

  filtered = filtered.filter(w => !(driftBlock.has(w) && !contextLC.includes(w)));
  if (filtered.length === 0) return null;

  // Hard clamp to 2 words max
  filtered = filtered.slice(0, 2);
  a = filtered.join(' ');
  if (!/^[a-z ]{1,40}$/.test(a)) return null;
  return a;
}

function dedupeLowerLoose(arr: string[]): string[] {
  const seen = new Set();
  return arr.filter(a => {
    const key = a.toLowerCase().trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function extractArray(raw: string): any[] {
  try { const p = JSON.parse(raw); if (Array.isArray(p)) return p; } catch {}
  const m = raw.match(/\[[\s\S]*\]/);
  if (m) { try { const p = JSON.parse(m[0]); if (Array.isArray(p)) return p; } catch {} }
  return [];
}

// ------------------- Main Function -------------------
export async function generateFlashcards(
  context: string,
  requestedCount: number,
  maxCount: number,
  promptVersion: number,
  answerLength: string = 'mixed',
  recentAvg: number | null = null,
  options: FlashcardGenOptions = {},
  availableSymbols: string[] = [] // optional
) {
  context = trimContext(context);
  const contextLC = context.toLowerCase();

  const relatedPrompts = dedupeLowerLoose((options.relatedPrompts || [])
    .filter(p => p && p.trim().length > 0 && p.trim().toLowerCase() !== contextLC)
    .slice(0, 5));

  let previousAnswers = dedupeLowerLoose((options.previousAnswers || [])
    .map(a => clampShortRaw(a))
    .filter(a => a.length > 0)
    .slice(0, 50));

  // Filter previous answers by semantic relevance to context, not by hardcoded categories
  const contextTokens = new Set(contextLC.split(/\s+/).filter(w => w.length > 2));
  previousAnswers = previousAnswers.filter(pa => {
    const paTokens = pa.split(/\s+/);
    // Keep if: answer overlaps with context OR is a common AAC connector word
    return paTokens.some(pt => contextTokens.has(pt)) || /^(i|you|me|we|and|or|but|is|are|have|want|need|like|love|help|more|stop|go|yes|no)$/.test(pa);
  });

  const driftBlock = new Set(options.driftBlock || []);
  const symbolList = availableSymbols.slice(0, 150).map(s => s.replace('.svg','')).join(', ');

  // ------------------- Build prompt -------------------
  const prompt = `
You are generating AAC vocabulary flashcards.

TASK:
Given the USER PROMPT text, choose the best 1-2 word responses that the user might want to say.

RULES:
- Answers MUST be 1 word whenever possible. Use 2 words only if absolutely necessary.
- Each answer MUST be directly usable as something the user could tap/say in response.
- Do NOT explain, describe, or use full sentences.
- Use only plain words in English, no punctuation.
- If you cannot think of enough good answers, return fewer items. Quality over quantity.

${symbolList ? `AVAILABLE VOCABULARY (prefer these when relevant):\n${symbolList}\n\n` : ''}

FORMAT:
Return ONLY a valid JSON object, no markdown, like:
{
  "cards": [
    {"question": "${context}", "answer": "<word1>"},
    {"question": "${context}", "answer": "<word2>"}
  ]
}

USER PROMPT: "${context}"

Generate up to ${requestedCount} answers, ordered by most useful first.
`.trim();

  const modelUsed = process.env.FAST_MODEL_NAME || 'gpt-5-nano';
  const temperature = Number(process.env.GEN_TEMPERATURE ?? 0.1);
  const openai = getOpenAI();

  const completion = await openai.chat.completions.create({
    model: modelUsed,
    temperature,
    response_format: { type: 'json_object' },
    messages: [
      { 
        role: 'system', 
        content: 'You are an AAC assistant. Output only JSON as specified by the user. Each answer must be 1-2 plain English words usable as a button label. Do not add explanations, notes, or extra text.'
      },
      { role: 'user', content: prompt }
    ]
  });

  const raw = completion.choices?.[0]?.message?.content || '{"cards":[]}';
  let arr: any[] = [];
  try {
    const obj = JSON.parse(raw);
    arr = Array.isArray(obj.cards) ? obj.cards : [];
  } catch {
    arr = extractArray(raw) as any[];
  }

  const seen = new Set<string>();
  const cards: Flashcard[] = [];

  for (const o of arr) {
    if (!o || typeof o.answer !== 'string') continue;
    let cleaned = cleanAnswer(o.answer, contextLC, driftBlock, !!options.strictSymbols);
    if (!cleaned) continue;
    if (seen.has(cleaned)) continue;
    seen.add(cleaned);
    cards.push({ question: context, answer: cleaned });
    if (cards.length >= requestedCount) break;
  }

  // Minimal fallback: only if critically under-producing and have available symbols
  if (cards.length < requestedCount * 0.15 && !options.strictSymbols && availableSymbols.length > 0) {
    const availableSymbolNames = availableSymbols.map(s => s.replace('.svg','').toLowerCase());
    for (const sym of availableSymbolNames) {
      if (cards.length >= requestedCount) break;
      if (!seen.has(sym)) {
        seen.add(sym);
        cards.push({ question: context, answer: sym });
      }
    }
  }

  if (cards.length > requestedCount) cards.length = requestedCount;

  console.log('[GEN FILTER STATS]', {
    requested: requestedCount,
    produced: cards.length,
    fallbackUsed: cards.length < arr.length ? 'yes' : 'no',
    driftBlocked: Array.from(driftBlock).slice(0, 8)
  });

  return { cards, rawContent: raw, modelUsed };
}
