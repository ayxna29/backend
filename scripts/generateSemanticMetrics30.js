import fs from 'node:fs';

// Example config for 30 rows
const NUM_ROWS = 30;
const core_vocab = new Set([
  'i','you','me','it','want','go','stop','help','more','yes','no','please','thank you','mine','your','his','her','their','our','this','that','here','there','now','eat','drink','play','work','school','home','park','see','look','watch','show','give','get','find','make','open','close','turn','move','sit','stand','run','walk','come','leave','start','finish','again','first','next','last','before','after','big','small','happy','sad','angry','tired','hungry','thirsty','sick','hurt','safe','danger','fun','boring','busy','quiet','loud','fast','slow','hot','cold','clean','dirty','full','empty','ready','wait','up','down','in','out','on','off','was','could','by','as','if','or','dad','your','today','going','find','say','work','before','after','time','now','very','him','please','her','different','away','good','we','be','have','am','how','but','why','they','know','of','there','come','did','tell','at','down','big','little','could','give','by','as','if','or','dad','your','today','going','find','say','work','before','after','time','now','very','him','please','her'
]);
const regulation_words = new Set(['stop','quiet','help','home','done']);
const politeness_words = new Set(['thank you','please','okay','maybe']);

function randomWords(n, pool) {
  const arr = Array.from(pool);
  const out = [];
  for (let i = 0; i < n; i++) out.push(arr[Math.floor(Math.random() * arr.length)]);
  return out;
}

function randomExpected(n) {
  // Simulate expected words for each row
  return randomWords(n, core_vocab);
}

function randomModelOutput(n) {
  // Simulate model output (mix of core, regulation, politeness, and some drift)
  const all = [...core_vocab, ...regulation_words, ...politeness_words, 'banana','rocket','unicorn','abstract','philosophy'];
  return randomWords(n, all);
}

function randomLatency() {
  return Math.floor(Math.random() * 300) + 250;
}

const lines = [
  'expected\tmodel_output\tavg_latency_ms\texpected_match_score\tcore_vocab_score\tpersona_alignment\ttap_efficiency\tregulation_score\tdrift_rate'
];

for (let i = 1; i <= NUM_ROWS; i++) {
  const expected = randomExpected(3 + Math.floor(Math.random()*2));
  const model_output = randomModelOutput(4 + Math.floor(Math.random()*2));
  const latency = randomLatency();

  // 1. Expected match score
  const expected_match = expected.filter(w => model_output.includes(w)).length / expected.length;

  // 2. Core vocab compliance
  const core_vocab_score = model_output.filter(w => core_vocab.has(w)).length / model_output.length;

  // 3. Persona appropriateness (simulate: penalize if avg words > 2 or if abstract words)
  const avg_words = model_output.reduce((sum, w) => sum + w.split(' ').length, 0) / model_output.length;
  const abstraction_penalty = model_output.some(w => ['abstract','philosophy','unicorn','rocket'].includes(w)) ? 0.2 : 0;
  const verbosity_penalty = avg_words > 2 ? 0.2 : 0;
  const persona_alignment = 1 - (abstraction_penalty + verbosity_penalty);

  // 4. Tap efficiency
  const tap_efficiency = expected.length / model_output.length;

  // 5. Regulation score
  const regulation_score = model_output.filter(w => regulation_words.has(w)).length / model_output.length;

  // 6. Drift rate
  const drift_rate = model_output.filter(w => !core_vocab.has(w) && !expected.includes(w)).length / model_output.length;

  lines.push([
    JSON.stringify(expected),
    JSON.stringify(model_output),
    latency,
    expected_match.toFixed(2),
    core_vocab_score.toFixed(2),
    persona_alignment.toFixed(2),
    tap_efficiency.toFixed(2),
    regulation_score.toFixed(2),
    drift_rate.toFixed(2)
  ].join('\t'));
}

fs.writeFileSync('scripts/semantic_metrics_30.tsv', lines.join('\n'), 'utf8');
console.log('Generated scripts/semantic_metrics_30.tsv with 30 rows of all metrics.');
