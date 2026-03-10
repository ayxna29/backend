import 'dotenv/config';

import fs from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import path from 'node:path';

import { generateFlashcards } from '../src/generateFlashcards.js';
import Papa from 'papaparse';

interface Args {
  input?: string;
  out?: string;
  count: number;
  limit: number;
  promptVersion: number;
  delayMs: number;
  includeRaw: boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    count: 30,
    limit: 500,
    promptVersion: 1,
    delayMs: 0,
    includeRaw: false
  };

  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === '--input' && val) { args.input = val; i += 1; continue; }
    if (key === '--out' && val) { args.out = val; i += 1; continue; }
    if (key === '--count' && val) { args.count = Number(val); i += 1; continue; }
    if (key === '--limit' && val) { args.limit = Number(val); i += 1; continue; }
    if (key === '--promptVersion' && val) { args.promptVersion = Number(val); i += 1; continue; }
    if (key === '--delayMs' && val) { args.delayMs = Number(val); i += 1; continue; }
    if (key === '--includeRaw') { args.includeRaw = true; continue; }
  }

  if (!Number.isFinite(args.count) || args.count < 1) args.count = 30;
  if (!Number.isFinite(args.limit) || args.limit < 1) args.limit = 500;
  if (!Number.isFinite(args.promptVersion) || args.promptVersion < 1) args.promptVersion = 1;
  if (!Number.isFinite(args.delayMs) || args.delayMs < 0) args.delayMs = 0;

  return args;
}

async function resolveInputPath(explicit?: string): Promise<string> {
  if (explicit) return explicit;
  const candidates = [
    'prompts.txt'
  ];

  for (const rel of candidates) {
    const full = path.resolve(process.cwd(), rel);
    try {
      await fs.access(full);
      return full;
    } catch {
      // ignore
    }
  }

  throw new Error('No input file found. Provide --input <path> (json/csv/txt).');
}

async function loadPrompts(inputPath: string): Promise<string[]> {
  const ext = path.extname(inputPath).toLowerCase();
  const raw = await fs.readFile(inputPath, 'utf8');

  if (ext === '.json') {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed.map(String);
    if (parsed && Array.isArray(parsed.prompts)) return parsed.prompts.map(String);
    throw new Error('JSON must be an array of strings or {"prompts": [...]}');
  }

  if (ext === '.csv') {
    const parsed = Papa.parse(raw, { header: true, skipEmptyLines: true });
    if (parsed.errors?.length) {
      throw new Error(`CSV parse error: ${parsed.errors[0].message}`);
    }
    const rows = parsed.data as Record<string, string>[];
    if (rows.length === 0) return [];
    const firstRow = rows[0];
    const promptKey = Object.keys(firstRow).find(k => k.toLowerCase() === 'prompt');
    if (promptKey) return rows.map(r => String(r[promptKey] ?? '')).filter(Boolean);
    // fallback: first column value
    const firstKey = Object.keys(firstRow)[0];
    return rows.map(r => String(r[firstKey] ?? '')).filter(Boolean);
  }

  // default: txt (one prompt per line)
  return raw
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = await resolveInputPath(args.input);
  const prompts = await loadPrompts(inputPath);
  const limited = prompts.slice(0, args.limit);

  if (limited.length === 0) {
    throw new Error('No prompts found in input file.');
  }

  const outPath = args.out
    ? path.resolve(process.cwd(), args.out)
    : path.resolve(process.cwd(), 'scripts/prompt-test-results.jsonl');

  const outStream = createWriteStream(outPath, { encoding: 'utf8' });

  for (let i = 0; i < limited.length; i += 1) {
    const prompt = String(limited[i] ?? '').trim();
    if (!prompt) continue;
    try {
      const result = await generateFlashcards(
        prompt,
        args.count,
        args.count,
        args.promptVersion,
        'mixed',
        null,
        {},
        []
      );

      const payload: Record<string, unknown> = {
        index: i,
        prompt,
        cards: result.cards,
        modelUsed: result.modelUsed
      };

      if (args.includeRaw) payload.rawContent = result.rawContent;

      outStream.write(`${JSON.stringify(payload)}\n`);
    } catch (err) {
      outStream.write(`${JSON.stringify({ index: i, prompt, error: String(err) })}\n`);
    }

    if (args.delayMs > 0 && i < limited.length - 1) {
      await sleep(args.delayMs);
    }
  }

  outStream.end();

  console.log(`Done. Wrote ${limited.length} results to ${outPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
