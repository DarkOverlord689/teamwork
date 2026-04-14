export const SPEAKER_PALETTE = [
  '#1976D2', // blue (primary)
  '#9C27B0', // purple (secondary)
  '#2E7D32', // green (success)
  '#ED6C02', // orange (warning)
  '#D32F2F', // red (error)
  '#0288D1', // light blue (info)
  '#6A1B9A', // deep purple
  '#4E342E', // brown
];

export const EMOTION_COLOR_MAP: Record<string, string> = {
  neutral: '#9E9E9E',
  happy: '#4CAF50',
  sad: '#2196F3',
  angry: '#F44336',
  surprised: '#FFEB3B',
  fear: '#9C27B0',
  disgust: '#795548',
};

export function hashSpeakerId(speakerId: string): number {
  // djb2 hash
  let hash = 5381;
  for (let i = 0; i < speakerId.length; i++) {
    hash = ((hash << 5) + hash) + speakerId.charCodeAt(i);
    hash = hash & hash; // 32-bit int
  }
  return Math.abs(hash);
}

export function getSpeakerColor(speakerId: string): string {
  return SPEAKER_PALETTE[hashSpeakerId(speakerId) % SPEAKER_PALETTE.length];
}

export function getEmotionColor(emotion: string): string {
  return EMOTION_COLOR_MAP[emotion] ?? '#9E9E9E';
}
