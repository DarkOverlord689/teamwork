// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { downsampleVisionFrames } from '../utils/downsample';
import type { FrameResult } from '../../../types/audit';

const makeFrame = (
  timestamp: number,
  personId: string,
  emotion: string | null,
  isLooking: boolean | null
): FrameResult => ({
  frame_number: Math.round(timestamp * 30),
  timestamp,
  persons: [{
    person_id: personId,
    bbox: { x1: 0, y1: 0, x2: 100, y2: 100 },
    gaze: isLooking !== null ? {
      direction: { yaw: 0, pitch: 0, roll: 0 },
      is_looking_at_camera: isLooking,
      confidence: 0.9,
      category: 'front',
    } : null,
    gesture: null,
    pose: null,
    emotion: emotion !== null ? {
      primary_emotion: emotion,
      confidence: 0.8,
      all_emotions: { [emotion]: 0.8 },
    } : null,
  }],
});

describe('downsampleVisionFrames', () => {
  it('returns empty array for empty input', () => {
    expect(downsampleVisionFrames([], 0.5)).toEqual([]);
  });

  it('groups 3 frames in same window into 1 VisionWindow with dominant emotion', () => {
    const frames = [
      makeFrame(0.1, 'P1', 'happy', true),
      makeFrame(0.2, 'P1', 'happy', true),
      makeFrame(0.3, 'P1', 'neutral', false),
    ];
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows).toHaveLength(1);
    expect(windows[0].dominant_emotion).toBe('happy');
    expect(windows[0].person_id).toBe('P1');
    expect(windows[0].frame_count).toBe(3);
  });

  it('treats null emotion as neutral', () => {
    const frames = [makeFrame(0.1, 'P1', null, false)];
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows[0].dominant_emotion).toBe('neutral');
  });

  it('treats null gaze as not looking at camera', () => {
    const frames = [makeFrame(0.1, 'P1', 'happy', null)];
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows[0].is_looking_at_camera).toBe(false);
  });

  it('uses majority vote for is_looking_at_camera', () => {
    const frames = [
      makeFrame(0.1, 'P1', 'happy', true),
      makeFrame(0.2, 'P1', 'happy', true),
      makeFrame(0.3, 'P1', 'happy', false),
    ];
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows[0].is_looking_at_camera).toBe(true);
  });

  it('produces 2 windows for frames spanning 2 buckets', () => {
    const frames = [
      makeFrame(0.1, 'P1', 'happy', true),   // bucket 0 [0, 0.5)
      makeFrame(0.6, 'P1', 'sad', false),     // bucket 1 [0.5, 1.0)
    ];
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows).toHaveLength(2);
  });

  it('processes frame with all-null optional fields without throwing', () => {
    const frames: FrameResult[] = [{
      frame_number: 1,
      timestamp: 0.1,
      persons: [{
        person_id: 'P1',
        bbox: { x1: 0, y1: 0, x2: 50, y2: 50 },
        gaze: null,
        gesture: null,
        pose: null,
        emotion: null,
      }],
    }];
    expect(() => downsampleVisionFrames(frames, 0.5)).not.toThrow();
    const windows = downsampleVisionFrames(frames, 0.5);
    expect(windows[0].dominant_emotion).toBe('neutral');
    expect(windows[0].is_looking_at_camera).toBe(false);
  });
});
