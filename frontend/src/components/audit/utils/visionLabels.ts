/**
 * Labels de traducción para los valores de análisis de visión
 * Los valores vienen en inglés desde el backend y se traducen aquí
 */

// Emociones detectadas por el modelo de visión
export const EMOTION_LABELS: Record<string, string> = {
  attentive: 'Atento',
  happy: 'Feliz',
  sad: 'Triste',
  angry: 'Enojado',
  surprised: 'Sorprendido',
  fearful: 'Temeroso',
  disgusted: 'Disgustado',
  neutral: 'Neutral',
  confused: 'Confundido',
};

// Direcciones de atención
export const ATTENTION_LABELS: Record<string, string> = {
  camera: 'Cámara',
  other_person: 'Otra persona',
  downward: 'Hacia abajo',
  away: 'Alejado',
  unclear: 'No claro',
};

// Niveles de engagement
export const ENGAGEMENT_LABELS: Record<string, string> = {
  high: 'Alto',
  medium: 'Medio',
  low: 'Bajo',
  unclear: 'No claro',
};

// Tipos de momento
export const MOMENT_TYPE_LABELS: Record<string, string> = {
  speech_onset: 'Inicio de habla',
  receiving_question: 'Recibiendo pregunta',
  hesitation: 'Hesitación',
  interruption_received: 'Interrupción recibida',
  post_question_silence: 'Silencio post-pregunta',
  end_of_long_turn: 'Fin de turno largo',
  back_channel: 'Retroalimentación',
};

/**
 * Función auxiliar para traducir valores
 * @param value - Valor en inglés del backend
 * @param dictionary - Diccionario de traducción
 * @returns Valor traducido o el valor original si no existe traducción
 */
export function translateLabel(value: string | undefined, dictionary: Record<string, string>): string {
  if (!value) return '—';
  return dictionary[value.toLowerCase()] || value;
}
