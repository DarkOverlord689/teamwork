export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  speaker_id: string;
  text: string;
  words: WordTimestamp[];
  language: string;
  no_speech_prob: number;
}

export interface SpeakerSegment {
  start: number;
  end: number;
  speaker_id: string;
  confidence: number;
}

export interface SpeakerTurn {
  start: number;
  end: number;
  speaker_id: string;
  duration: number;
  segment_count: number;
}

export interface Interruption {
  time: number;
  interrupter_id: string;
  interrupted_id: string;
  overlap_duration: number;
  interruption_type: 'disruptive' | 'back_channel' | 'cooperative';
}

export interface SpeakerMetrics {
  speaker_id: string;
  speaking_time_seconds: number;
  turn_count: number;
  interruption_count: number;
  interrupted_count: number;
  avg_turn_duration: number;
  participation_ratio: number;
  back_channel_count: number;
}

export interface AudioSessionMetrics {
  total_speakers: number;
  duration: number;
  total_speaking_time: number;
  silence_ratio: number;
  participation_cv: number;
  turn_alternation_rate: number;
  per_speaker_metrics: SpeakerMetrics[];
}

export interface AudioResult {
  video_path: string;
  duration_seconds: number;
  sample_rate: number;
  segments: SpeakerSegment[];
  turns: SpeakerTurn[];
  transcripts: TranscriptSegment[];
  interruptions: Interruption[];
  processing_time_seconds: number;
  session_metrics: AudioSessionMetrics;
}

export interface GazeDetail {
  /** [yaw, pitch, roll] in degrees – serialised as a list by the backend */
  direction: [number, number, number];
  is_looking_at_camera: boolean;
  confidence: number;
  category: string;
}

export interface GestureDetail {
  gesture_type: string;
  confidence: number;
  intensity: number;
}

export interface PoseDetail {
  body_orientation: number;
  shoulder_angle: number;
  confidence: number;
}

export interface EmotionDetail {
  primary_emotion: string;
  confidence: number;
  all_emotions: Record<string, number>;
}

export interface PersonFrame {
  person_id: string;
  /** [x, y, w, h] in pixels – serialised as a list by the backend */
  bbox: [number, number, number, number];
  gaze: GazeDetail | null;
  gesture: GestureDetail | null;
  pose: PoseDetail | null;
  emotion: EmotionDetail | null;
}

export interface FrameResult {
  frame_number: number;
  timestamp: number;
  persons: PersonFrame[];
}

export interface PersonMetrics {
  person_id: string;
  total_frames_seen: number;
  gaze_contact_percentage: number;
  dominant_emotion: string;
  emotion_distribution: Record<string, number>;
  average_body_orientation: number;
  gesture_counts: Record<string, number>;
  attention_score: number;
}

export interface VisionSessionMetrics {
  total_persons: number;
  total_frames: number;
  duration: number;
  per_person_metrics: PersonMetrics[];
}

export interface VisionResult {
  video_path: string;
  total_frames: number;
  fps_processed: number;
  duration_seconds: number;
  processing_time_seconds: number;
  frames: FrameResult[];
  person_embeddings: Record<string, number[]>;
  session_metrics: VisionSessionMetrics;
  /** Mapping of string timestamp (e.g. "2.0") to relative API URL for the JPEG thumbnail */
  frame_thumbnails?: Record<string, string>;
  /** Whether this result was produced by the smart frame selection pipeline */
  smart_mode?: boolean;
  /** Qwen/multimodal key moment analyses (populated when smart_mode=true) */
  key_moment_analyses?: MultimodalFrameAnalysis[];
}

export interface MultimodalFrameAnalysis {
  frame_number: number;
  timestamp_ms: number;
  speaker_id: string | null;
  person_id: string | null;
  moment_type: string | null;
  multimodal_description: string | null;
  emotion_primary: string | null;
  emotion_confidence: number | null;
  attention_direction: string | null;
  engagement_level: string | null;
  face_detected: boolean;
  fallback_model: string | null;
  fallback_reason: string | null;
  /** Relative URL to the JPEG frame served by the backend, e.g. /api/v1/vision/frames/{session_id}/frame_2381.jpg */
  image_url?: string | null;
}

export interface VisionWindow {
  windowStart: number;
  windowEnd: number;
  person_id: string;
  dominant_emotion: string;
  is_looking_at_camera: boolean;
  frame_count: number;
  /** Absolute URL of the nearest saved frame thumbnail, if available */
  frame_url?: string;
  /** Qwen/multimodal natural language description for this window (smart mode) */
  multimodal_description?: string | null;
  /** Primary emotion from Qwen analysis (smart mode) */
  emotion_primary?: string | null;
  /** Emotion confidence from Qwen analysis (0-1) */
  emotion_confidence?: number | null;
  /** Attention direction from Qwen analysis */
  attention_direction?: string | null;
  /** Engagement level from Qwen analysis */
  engagement_level?: string | null;
  /** Moment type from smart frame selection */
  moment_type?: string | null;
  /** Speaker ID associated with this window (smart mode) */
  speaker_id?: string | null;
  /** Whether this window came from a smart/multimodal key moment analysis */
  is_smart_window?: boolean;
  /** Direct image URL from key_moment_analyses (smart mode), e.g. /api/v1/vision/frames/{session_id}/frame_2381.jpg */
  image_url?: string | null;
}
