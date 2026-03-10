import fs from 'node:fs';

// Example symbol set for matching (replace with your actual set)
const SYMBOLS = new Set([
  'yes', 'no', 'more', 'please', 'help', 'want', 'go', 'stop', 'thank you', 'okay', 'good', 'bad', 'I', 'you', 'me', 'it', 'where', 'who', 'what', 'when', 'why', 'how', 'mine', 'your', 'his', 'her', 'their', 'our', 'this', 'that', 'here', 'there', 'now', 'later', 'up', 'down', 'in', 'out', 'on', 'off',
  'eat', 'drink', 'play', 'work', 'school', 'home', 'park', 'read', 'write', 'see', 'look', 'watch', 'show', 'give', 'get', 'find', 'make', 'open', 'close', 'turn', 'move', 'sit', 'stand', 'run', 'walk', 'come', 'leave', 'start', 'finish', 'again', 'first', 'next', 'last', 'before', 'after',
  'big', 'small', 'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty', 'sick', 'hurt', 'safe', 'danger', 'fun', 'boring', 'busy', 'quiet', 'loud', 'fast', 'slow', 'hot', 'cold', 'clean', 'dirty', 'full', 'empty', 'ready', 'wait',
  'goes', 'let', "let's", 'call', 'name', 'baby', 'cute', 'hold', 'together', 'maybe', 'almost', 'not yet', 'finished', 'try', 'explain', 'need', 'right', 'left', 'straight', 'back', 'store',
  'snack', 'lunch', 'breakfast', 'meal', 'food', 'nothing', 'playing', 'reading', 'working', 'eating', 'watching', 'fine', 'great', 'affirmative', 'sure', 'hi', 'hello', "what's up", 'howdy', 'not',
]);

// Core vocabulary for coverage (replace with your actual core set)
const CORE_VOCAB = new Set([
  'I', 'you', 'me', 'it', 'want', 'go', 'stop', 'help', 'more', 'yes', 'no', 'please', 'thank you', 'mine', 'your', 'his', 'her', 'their', 'our', 'this', 'that', 'here', 'there', 'now', 'eat', 'drink', 'play', 'work', 'school', 'home', 'park', 'see', 'look', 'watch', 'show', 'give', 'get', 'find', 'make', 'open', 'close', 'turn', 'move', 'sit', 'stand', 'run', 'walk', 'come', 'leave', 'start', 'finish', 'again', 'first', 'next', 'last', 'before', 'after', 'big', 'small', 'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty', 'sick', 'hurt', 'safe', 'danger', 'fun', 'boring', 'busy', 'quiet', 'loud', 'fast', 'slow', 'hot', 'cold', 'clean', 'dirty', 'full', 'empty', 'ready', 'wait'
]);

function evaluate(entries) {
  const results = [];
  for (const entry of entries) {
    const answers = entry.cards.map(c => c.answer.toLowerCase());
    // Symbol accuracy: percent of answers in SYMBOLS
    const symbolMatches = answers.filter(a => SYMBOLS.has(a)).length;
    const symbolAccuracy = answers.length ? symbolMatches / answers.length : 0;
    // Core coverage: percent of answers in CORE_VOCAB
    const coreMatches = answers.filter(a => CORE_VOCAB.has(a)).length;
    const coreCoverage = answers.length ? coreMatches / answers.length : 0;
    // Avg tap efficiency: unique answers / total answers
    const uniqueAnswers = new Set(answers).size;
    const avgTapEff = answers.length ? uniqueAnswers / answers.length : 0;
    // Latency: use entry.latencyMs if available, else null
    results.push({
      statement: entry.prompt,
      avg_latency_ms: entry.latencyMs ?? null,
      symbol_accuracy: +(symbolAccuracy * 100).toFixed(2),
      core_coverage: +(coreCoverage * 100).toFixed(2),
      avg_tap_eff: +avgTapEff.toFixed(2)
    });
  }
  return results;
}

function main() {
  const lines = fs.readFileSync('scripts/prompt-test-results.jsonl', 'utf8').split(/\r?\n/).filter(Boolean);
  const entries = lines.map(l => JSON.parse(l));
  // Only evaluate entries with index >= 100
  const filtered = entries.filter(e => e.index >= 100);
  const evalResults = evaluate(filtered);
  // Output as TSV
  console.log('statement\tavg_latency_ms\tsymbol_accuracy\tcore_coverage\tavg_tap_eff');
  for (const r of evalResults) {
    console.log(`${r.statement}\t${r.avg_latency_ms ?? ''}\t${r.symbol_accuracy}%\t${r.core_coverage}%\t${r.avg_tap_eff}`);
  }
}

main();
