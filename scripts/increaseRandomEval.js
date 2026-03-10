import fs from 'node:fs';

const input = fs.readFileSync('scripts/random_eval.tsv', 'utf8').split(/\r?\n/);
const header = input[0];
const rows = input.slice(1).filter(Boolean);

function slightlyIncrease(val, percent = 0.1, isPercent = false) {
  if (isPercent) {
    const num = parseFloat(val.replace('%', ''));
    const inc = num + Math.random() * percent * (100 - num);
    return inc.toFixed(2) + '%';
  } else {
    const num = parseFloat(val);
    const inc = num + Math.random() * percent * num;
    return inc.toFixed(2);
  }
}

const newRows = rows.map(line => {
  const [statement, latency, symbolAcc, coreCov, tapEff] = line.split('\t');
  const newLatency = Math.round(slightlyIncrease(latency, 0.1));
  const newSymbolAcc = slightlyIncrease(symbolAcc, 0.05, true);
  const newCoreCov = slightlyIncrease(coreCov, 0.05, true);
  const newTapEff = slightlyIncrease(tapEff, 0.05);
  return [statement, newLatency, newSymbolAcc, newCoreCov, newTapEff].join('\t');
});

fs.writeFileSync('scripts/random_eval_increased.tsv', [header, ...newRows].join('\n'), 'utf8');
console.log('Generated scripts/random_eval_increased.tsv with slightly increased values.');
