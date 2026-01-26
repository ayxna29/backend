import { z } from 'zod';

export const MAX_CONTEXT_CHARS = 4000;
export const MAX_COUNT = 15;
export const MAX_TAG_LEN = 64;

const toTrimmed = (s: unknown) => (typeof s === 'string' ? s.trim() : '');
const toInt = (v: unknown) => {
  if (typeof v === 'number' && Number.isInteger(v)) return v;
  if (typeof v === 'string' && /^[0-9]+$/.test(v)) return parseInt(v, 10);
  return v;
};

export const GenerateBodySchema = z.object({
  context: z
    .preprocess(toTrimmed, z.string().min(10).max(MAX_CONTEXT_CHARS)),
  tag: z
    .preprocess(toTrimmed, z.string().min(1).max(MAX_TAG_LEN))
    .optional(),
  count: z
    .preprocess(toInt, z.number().int().min(1).max(MAX_COUNT))
    .optional()
});

export type GenerateBody = z.infer<typeof GenerateBodySchema>;

export const FlashcardSchema = z.object({
  question: z.preprocess(toTrimmed, z.string().min(3).max(300)),
  answer: z.preprocess(toTrimmed, z.string().min(1).max(800))
});

export interface CleanStats {
  cards: { question: string; answer: string }[];
  rejected: number;
  duplicates: number;
}

export function cleanFlashcards(raw: any[]): CleanStats {
  const cards: { question: string; answer: string }[] = [];
  let rejected = 0;
  let duplicates = 0;
  const seen = new Set<string>();

  for (const c of (Array.isArray(raw) ? raw : [])) {
    const parsed = FlashcardSchema.safeParse({
      question: c?.question,
      answer: c?.answer
    });
    if (!parsed.success) {
      rejected++;
      continue;
    }
    const qNorm = parsed.data.question.toLowerCase().replace(/\s+/g, ' ');
    if (seen.has(qNorm)) {
      duplicates++;
      continue;
    }
    seen.add(qNorm);
    cards.push({
      question: parsed.data.question,
      answer: parsed.data.answer
    });
  }
  return { cards, rejected, duplicates };
}

export function parseGenerateBody(body: unknown, defaultCount = 5): GenerateBody {
  const parsed = GenerateBodySchema.safeParse(body || {});
  if (!parsed.success) {
    const first = parsed.error.issues[0];
    throw new Error(first?.message || 'Invalid body');
  }
  return {
    ...parsed.data,
    count: parsed.data.count ?? defaultCount
  };
}

export function normalizeTag(tag?: string): string | undefined {
  if (!tag) return;
  const t = tag.trim();
  if (!t) return;
  return t.slice(0, MAX_TAG_LEN);
}

// Helper to validate an array of arbitrary raw cards (e.g. from model)
export function validateRawCards(raw: unknown): { cards: { question: string; answer: string }[]; errors: number } {
  if (!Array.isArray(raw)) return { cards: [], errors: 1 };
  const cards: { question: string; answer: string }[] = [];
  let errors = 0;
  for (const r of raw) {
    const p = FlashcardSchema.safeParse(r);
    if (!p.success) {
      errors++;
      continue;
    }
    cards.push(p.data);
  }
  return { cards, errors };
}