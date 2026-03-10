import fs from 'node:fs';

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomPercent(min, max) {
  return (Math.random() * (max - min) + min).toFixed(2) + '%';
}

function randomFloat(min, max, decimals = 2) {
  return (Math.random() * (max - min) + min).toFixed(decimals);
}

const lines = ['statement\tavg_latency_ms\tsymbol_accuracy\tcore_coverage\tavg_tap_eff'];
for (let i = 1; i <= 463; i++) {
  const statement = `statement_${i}`;
  const avg_latency_ms = randomInt(200, 800);
  const symbol_accuracy = randomPercent(50, 100);
  const core_coverage = randomPercent(30, 100);
  const avg_tap_eff = randomFloat(0.4, 1.0);
  lines.push(`${statement}\t${avg_latency_ms}\t${symbol_accuracy}\t${core_coverage}\t${avg_tap_eff}`);
}

fs.writeFileSync('scripts/random_eval.tsv', lines.join('\n'), 'utf8');
console.log('Generated scripts/random_eval.tsv with 463 random rows.');
