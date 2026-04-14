export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

export function formatMs(ms: number): string {
  return `${(ms / 1000).toFixed(2)}s`;
}

export function tickInterval(durationSeconds: number): number {
  if (durationSeconds <= 60) return 5;
  if (durationSeconds <= 300) return 15;
  if (durationSeconds <= 1200) return 60;
  return 120;
}
