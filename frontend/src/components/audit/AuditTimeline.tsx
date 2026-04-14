import { useEffect, useRef, useState, useMemo } from 'react';
import { Alert, Box, Button, LinearProgress, Typography } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../../store';
import { fetchAudioResult, fetchVisionResult } from '../../store/slices/transcriptAuditSlice';
import { getSpeakerColor } from './utils/speakerColor';
import { downsampleVisionFrames } from './utils/downsample';
import TimelineRuler from './TimelineRuler';
import SpeakerLane from './SpeakerLane';
import VisionLane from './VisionLane';

const LABEL_WIDTH = 80;

export default function AuditTimeline({ sessionId }: { sessionId: string }) {
  const dispatch = useDispatch<AppDispatch>();
  const session = useSelector((state: RootState) =>
    state.transcriptAudit.sessions[sessionId]
  );

  const audioResult = session?.audioResult ?? null;
  const visionResult = session?.visionResult ?? null;
  const loadingAudio = session?.loadingAudio ?? false;
  const loadingVision = session?.loadingVision ?? false;
  const errorAudio = session?.errorAudio ?? null;
  const errorVision = session?.errorVision ?? null;

  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  // Fetch on mount if not already loaded
  useEffect(() => {
    if (!audioResult && !loadingAudio && !errorAudio) {
      dispatch(fetchAudioResult(sessionId));
    }
    if (!visionResult && !loadingVision && !errorVision) {
      dispatch(fetchVisionResult(sessionId));
    }
  }, [sessionId]);

  // ResizeObserver for responsive timeline width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect?.width;
      if (w) setContainerWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const durationSeconds =
    audioResult?.session_metrics?.duration ??
    audioResult?.duration_seconds ??
    0;

  const pixelsPerSecond = durationSeconds > 0
    ? Math.max(1, (containerWidth - LABEL_WIDTH) / durationSeconds)
    : 10;

  const visionWindows = useMemo(() => {
    if (!visionResult) return [];
    // Use a larger window in smart mode (key moments are sparse) or for large frame counts
    const windowSize = visionResult.smart_mode ? 1.0 : ((visionResult.total_frames ?? 0) > 100_000 ? 2.0 : 0.5);
    return downsampleVisionFrames(
      visionResult.frames ?? [],
      windowSize,
      visionResult.frame_thumbnails,
      visionResult.key_moment_analyses,
    );
  }, [visionResult]);

  const speakerIds = useMemo(() =>
    Array.from(new Set((audioResult?.turns ?? []).map(t => t.speaker_id))),
    [audioResult]
  );

  const personIds = useMemo(() => {
    const fromMetrics = (visionResult?.session_metrics?.per_person_metrics ?? []).map(m => m.person_id);
    // In smart mode, also include person IDs from key_moment_analyses (Qwen/CLIP results).
    // Fall back to 'unknown' so that a lane is always rendered when key moments exist
    // even when person_id and speaker_id are both null (e.g. pure CLIP fallback runs).
    const smartAnalyses = visionResult?.key_moment_analyses ?? [];
    const fromSmartAnalyses = smartAnalyses.length > 0
      ? smartAnalyses.map(a => a.person_id ?? a.speaker_id ?? 'unknown')
      : [];
    return Array.from(new Set([...fromMetrics, ...fromSmartAnalyses]));
  }, [visionResult]);

  const noAudioData = !audioResult?.turns?.length && !audioResult?.interruptions?.length;
  const noVisionData = !visionResult?.frames?.length
    && !visionResult?.session_metrics?.per_person_metrics?.length
    && !visionResult?.key_moment_analyses?.length;

  return (
    <Box ref={containerRef} sx={{ p: 2 }}>
      {(loadingAudio || loadingVision) && <LinearProgress sx={{ mb: 1 }} />}

      {errorAudio && (
        <Alert severity="error" sx={{ mb: 1 }}
          action={<Button size="small" onClick={() => dispatch(fetchAudioResult(sessionId))}>Reintentar</Button>}>
          Error al cargar datos de audio.
        </Alert>
      )}
      {errorVision && (
        <Alert severity="error" sx={{ mb: 1 }}
          action={<Button size="small" onClick={() => dispatch(fetchVisionResult(sessionId))}>Reintentar</Button>}>
          Error al cargar datos de visión.
        </Alert>
      )}

      {durationSeconds === 0 && !loadingAudio && !errorAudio && (
        <Typography color="text.secondary" sx={{ p: 2 }}>
          No hay datos de análisis disponibles para esta sesión.
        </Typography>
      )}

      {durationSeconds > 0 && (
        <Box sx={{ overflowX: 'auto' }}>
          {/* Ruler */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ width: LABEL_WIDTH, flexShrink: 0 }} />
            <TimelineRuler durationSeconds={durationSeconds} pixelsPerSecond={pixelsPerSecond} />
          </Box>

          {/* Audio section */}
          <Typography variant="overline" sx={{ display: 'block', mt: 1, mb: 0.5, color: 'text.secondary' }}>
            Audio
          </Typography>
          {noAudioData && !loadingAudio && (
            <Typography variant="body2" color="text.secondary" sx={{ pl: 1 }}>
              Sin datos de turnos de audio para esta sesión.
            </Typography>
          )}
          {speakerIds.map(sid => (
            <Box key={sid} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <Box sx={{
                width: LABEL_WIDTH, flexShrink: 0, pr: 1,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
              }}>
                <Typography variant="caption" sx={{ color: getSpeakerColor(sid), fontWeight: 600 }}>
                  {sid}
                </Typography>
              </Box>
              <SpeakerLane
                speakerId={sid}
                turns={(audioResult?.turns ?? []).filter(t => t.speaker_id === sid)}
                interruptions={audioResult?.interruptions ?? []}
                durationSeconds={durationSeconds}
                pixelsPerSecond={pixelsPerSecond}
                color={getSpeakerColor(sid)}
              />
            </Box>
          ))}

          {/* Vision section */}
          <Typography variant="overline" sx={{ display: 'block', mt: 2, mb: 0.5, color: 'text.secondary' }}>
            Visión
          </Typography>
          {noVisionData && !loadingVision && (
            <Typography variant="body2" color="text.secondary" sx={{ pl: 1 }}>
              Sin datos de visión para esta sesión.
            </Typography>
          )}
          {personIds.map(pid => (
            <Box key={pid} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <Box sx={{
                width: LABEL_WIDTH, flexShrink: 0, pr: 1,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
              }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {pid}
                </Typography>
              </Box>
              <VisionLane
                personId={pid}
                windows={visionWindows}
                durationSeconds={durationSeconds}
                pixelsPerSecond={pixelsPerSecond}
              />
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}
