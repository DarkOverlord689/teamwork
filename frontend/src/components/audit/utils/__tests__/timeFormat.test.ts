import { describe, it, expect } from 'vitest';
import { formatTime, formatMs, tickInterval } from '../timeFormat';

describe('formatTime', () => {
  it('formats zero seconds', () => {
    expect(formatTime(0)).toBe('00:00.0');
  });

  it('formats 75.3 seconds', () => {
    expect(formatTime(75.3)).toBe('01:15.3');
  });

  it('formats 3600 seconds (1 hour)', () => {
    expect(formatTime(3600)).toBe('60:00.0');
  });

  it('pads single-digit minutes', () => {
    expect(formatTime(5)).toBe('00:05.0');
  });
});

describe('formatMs', () => {
  it('formats milliseconds as seconds with 2 decimals', () => {
    expect(formatMs(1500)).toBe('1.50s');
    expect(formatMs(250)).toBe('0.25s');
  });
});

describe('tickInterval', () => {
  it('returns 5 for short sessions (<=60s)', () => {
    expect(tickInterval(30)).toBe(5);
    expect(tickInterval(60)).toBe(5);
  });

  it('returns 15 for medium sessions (61-300s)', () => {
    expect(tickInterval(120)).toBe(15);
    expect(tickInterval(300)).toBe(15);
  });

  it('returns 60 for long sessions (301-1200s)', () => {
    expect(tickInterval(600)).toBe(60);
    expect(tickInterval(1200)).toBe(60);
  });

  it('returns 120 for very long sessions (>1200s)', () => {
    expect(tickInterval(1500)).toBe(120);
  });
});
