import OpenAI from 'openai';

let openaiClient: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (openaiClient) return openaiClient;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('Missing OPENAI_API_KEY');
  openaiClient = new OpenAI({ apiKey: key });
  return openaiClient;
}

export type SpeechTone = 'neutral' | 'happy' | 'excited' | 'sad' | 'frustrated';

const TONE_INSTRUCTIONS: Record<Exclude<SpeechTone, 'neutral'>, string> = {
  happy: 'Speak in a warm, happy, cheerful tone.',
  excited: 'Speak in an excited, upbeat, enthusiastic tone.',
  sad: 'Speak in a sad, gentle, subdued tone.',
  frustrated: 'Speak in a frustrated, urgent, insistent tone — like something needs attention right away.',
};

// Voices OpenAI's TTS API actually supports — kept in sync with the
// `voice` field type in node_modules/openai/resources/audio/speech.d.ts.
export const TTS_VOICES = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse'] as const;
export type TtsVoice = (typeof TTS_VOICES)[number];

const DEFAULT_VOICE: TtsVoice = 'nova';

/**
 * Synthesizes speech for `text` via OpenAI's TTS API and returns raw audio
 * bytes (mp3). Only `gpt-4o-mini-tts` supports the `instructions` field
 * used here for tone — that's a deliberate choice, not just the default
 * model, since tone is the whole point of this endpoint existing.
 */
export async function synthesizeSpeech(
  text: string,
  opts: { voice?: string; tone?: SpeechTone } = {}
): Promise<Buffer> {
  const voice = (TTS_VOICES as readonly string[]).includes(opts.voice || '')
    ? (opts.voice as TtsVoice)
    : DEFAULT_VOICE;
  const tone = opts.tone ?? 'neutral';
  const instructions = tone === 'neutral' ? undefined : TONE_INSTRUCTIONS[tone];

  const openai = getOpenAI();
  const response = await openai.audio.speech.create({
    model: 'gpt-4o-mini-tts',
    voice,
    input: text,
    response_format: 'mp3',
    ...(instructions ? { instructions } : {}),
  });
  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
