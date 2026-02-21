export function inferTagsFromCards(cards: {question:string; answer:string}[], max=5): string[] {
  const freq: Record<string, number> = {};
  const stop = new Set([
    'the','and','for','with','from','that','this','have','uses','into','over','what','how',
    'why','when','which','their','they','are','was','were','will','can','has','its','between' // keep domain tags? remove if you want these inferred
  ]);
  for (const c of cards) {
    const blob = `${c.question} ${c.answer}`.toLowerCase();
    for (const raw of blob.split(/\W+/)) {
      const w = raw.trim();
      if (w.length < 3) continue;
      if (stop.has(w)) continue;
      if (!/^[a-z0-9_-]+$/.test(w)) continue;
      freq[w] = (freq[w] || 0) + 1;
    }
  }
  return Object.entries(freq)
    .sort((a,b)=>b[1]-a[1])
    .slice(0, max)
    .map(([w])=>w);
}