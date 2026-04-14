import type { FrameResult, MultimodalFrameAnalysis, VisionWindow } from '../../../types/audit';

/**
 * Given the frame_thumbnails map (timestamp-string -> relative URL) from the
 * vision result, return the URL whose timestamp is closest to `targetSeconds`.
 * Returns undefined when the map is empty or undefined.
 */
function findNearestThumbnailUrl(
  thumbnails: Record<string, string> | undefined,
  targetSeconds: number,
): string | undefined {
  if (!thumbnails) return undefined;
  const entries = Object.entries(thumbnails);
  if (entries.length === 0) return undefined;

  let bestUrl: string | undefined;
  let bestDiff = Infinity;

  for (const [tsStr, url] of entries) {
    const diff = Math.abs(parseFloat(tsStr) - targetSeconds);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestUrl = url;
    }
  }

  return bestUrl;
}

export function downsampleVisionFrames(
  frames: FrameResult[],
  windowSize: number,
  frameThumbnails?: Record<string, string>,
  keyMomentAnalyses?: MultimodalFrameAnalysis[],
): VisionWindow[] {
  const windows: VisionWindow[] = [];

  // --- Smart mode: build windows from key_moment_analyses (Qwen results) ---
  if (keyMomentAnalyses && keyMomentAnalyses.length > 0) {
    for (const analysis of keyMomentAnalyses) {
      const timestampS = analysis.timestamp_ms / 1000.0;
      const windowStart = Math.floor(timestampS / windowSize) * windowSize;
      const windowEnd = windowStart + windowSize;
      const personId = analysis.person_id ?? analysis.speaker_id ?? 'unknown';

      // Prefer the direct image_url from the analysis; fall back to nearest thumbnail
      const image_url = analysis.image_url ?? null;
      const frame_url = image_url ?? findNearestThumbnailUrl(frameThumbnails, timestampS);

      windows.push({
        windowStart,
        windowEnd,
        person_id: personId,
        dominant_emotion: analysis.emotion_primary ?? 'neutral',
        is_looking_at_camera: analysis.attention_direction === 'camera',
        frame_count: 1,
        frame_url,
        image_url,
        multimodal_description: analysis.multimodal_description,
        emotion_primary: analysis.emotion_primary,
        emotion_confidence: analysis.emotion_confidence,
        attention_direction: analysis.attention_direction,
        engagement_level: analysis.engagement_level,
        moment_type: analysis.moment_type,
        speaker_id: analysis.speaker_id,
        is_smart_window: true,
      });
    }

    return windows.sort(
      (a, b) => a.windowStart - b.windowStart || a.person_id.localeCompare(b.person_id),
    );
  }

  // --- Standard mode: bucket regular frames into time windows ---
  if (!frames || frames.length === 0) return [];

  const buckets: Map<string, { personId: string; windowStart: number; windowEnd: number; emotions: string[]; gazeVotes: boolean[] }[]> = new Map();

  for (const frame of frames) {
    const bucketIdx = Math.floor(frame.timestamp / windowSize);
    const windowStart = bucketIdx * windowSize;
    const windowEnd = windowStart + windowSize;

    for (const person of frame.persons) {
      const key = `${person.person_id}__${bucketIdx}`;
      if (!buckets.has(key)) {
        buckets.set(key, [{
          personId: person.person_id,
          windowStart,
          windowEnd,
          emotions: [],
          gazeVotes: [],
        }]);
      }
      const bucket = buckets.get(key)![0];
      bucket.emotions.push(person.emotion?.primary_emotion ?? 'neutral');
      bucket.gazeVotes.push(person.gaze?.is_looking_at_camera ?? false);
    }
  }

  for (const [, [bucket]] of buckets) {
    // Modal emotion
    const emotionCounts: Record<string, number> = {};
    for (const e of bucket.emotions) emotionCounts[e] = (emotionCounts[e] ?? 0) + 1;
    const dominant_emotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'neutral';
    // Majority gaze
    const trueVotes = bucket.gazeVotes.filter(Boolean).length;
    const is_looking_at_camera = trueVotes > bucket.gazeVotes.length / 2;

    // Find the nearest thumbnail for the window midpoint
    const midpoint = (bucket.windowStart + bucket.windowEnd) / 2;
    const frame_url = findNearestThumbnailUrl(frameThumbnails, midpoint);

    windows.push({
      windowStart: bucket.windowStart,
      windowEnd: bucket.windowEnd,
      person_id: bucket.personId,
      dominant_emotion,
      is_looking_at_camera,
      frame_count: bucket.emotions.length,
      frame_url,
    });
  }

  return windows.sort((a, b) => a.windowStart - b.windowStart || a.person_id.localeCompare(b.person_id));
}
