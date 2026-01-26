import { createHash } from 'node:crypto';
import { getSupabase } from './supabase.js';
import { sha256Base64 } from './hash.js';

interface StartGenParams {
  userId: string | null;
  context: string;
  modelName: string;
  promptVersion: number;
  generationId: string; // new: externally supplied
}
export async function logGenerationStart(p: StartGenParams) {
  const supabase = getSupabase();
  const context_hash = sha256Base64(p.context);
  const { error } = await supabase
    .from('flashcard_generations')
    .insert({
      id: p.generationId,
      user_id: p.userId,
      context_text: p.context,
      context_hash,
      model_name: p.modelName,
      prompt_version: p.promptVersion,
      status: 'pending'
    });
  if (error) throw error;
  return { generationId: p.generationId };
}

interface CompleteGenParams {
  generationId: string;
  cards: { question: string; answer: string }[];
  raw: string | null;
  latencyMs: number;
  modelName: string;
  error?: string;
}
export async function logGenerationComplete(p: CompleteGenParams) {
  const supabase = getSupabase();
  const update = p.error
    ? {
        status: 'error',
        error_message: p.error,
        latency_ms: p.latencyMs,
        raw_response: p.raw,
        parsed_count: 0,
        model_name: p.modelName
      }
    : {
        status: 'success',
        error_message: null,
        latency_ms: p.latencyMs,
        raw_response: p.raw,
        parsed_count: p.cards.length,
        model_name: p.modelName
      };
  await supabase.from('flashcard_generations').update(update).eq('id', p.generationId);

  if (!p.error && p.cards.length) {
    const rows = p.cards.map((c, i) => ({
      generation_id: p.generationId,
      position: i,
      question: c.question,
      answer: c.answer
    }));
    await supabase.from('flashcard_generation_cards').insert(rows);
  }
}

interface FeedbackParams {
  userId: string;
  generationCardId: string;
  rating?: number;
  editedQuestion?: string;
  editedAnswer?: string;
  notes?: string;
}
export async function logFeedback(p: FeedbackParams) {
  const supabase = getSupabase();
  const { error } = await supabase.from('flashcard_feedback').insert({
    user_id: p.userId,
    generation_card_id: p.generationCardId,
    rating: p.rating ?? null,
    edited_question: p.editedQuestion ?? null,
    edited_answer: p.editedAnswer ?? null,
    notes: p.notes ?? null
  });
  if (error) throw error;
}