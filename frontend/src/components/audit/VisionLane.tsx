import { useState } from 'react';
import {
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { VisionWindow } from '../../types/audit';
import { getEmotionColor } from './utils/speakerColor';
import { formatTime } from './utils/timeFormat';
import {
  EMOTION_LABELS,
  ATTENTION_LABELS,
  ENGAGEMENT_LABELS,
  MOMENT_TYPE_LABELS,
  translateLabel,
} from './utils/visionLabels';

interface VisionLaneProps {
  personId: string;
  windows: VisionWindow[];
  durationSeconds: number;
  pixelsPerSecond: number;
  laneHeight?: number;
}

interface FrameModalProps {
  window: VisionWindow;
  personId: string;
  onClose: () => void;
}

function FrameModal({ window, personId, onClose }: FrameModalProps) {
  return (
    <Dialog open onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 1 }}>
        <span>
          {personId} — {formatTime(window.windowStart)} → {formatTime(window.windowEnd)}
        </span>
        <IconButton size="small" onClick={onClose} aria-label="Cerrar">
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {/* Prefer direct image_url (from key_moment_analyses), fall back to frame_url (thumbnail) */}
        {(window.image_url || window.frame_url) ? (
          <Box
            component="img"
            src={(window.image_url || window.frame_url) as string}
            alt={`Frame at ${formatTime(window.windowStart)}`}
            sx={{
              width: '100%',
              height: 'auto',
              display: 'block',
              borderRadius: 1,
              mb: 1.5,
              backgroundColor: 'black',
            }}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = 'none';
            }}
          />
        ) : (
          <Box
            sx={{
              width: '100%',
              height: 160,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'grey.100',
              borderRadius: 1,
              mb: 1.5,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Miniatura no disponible
            </Typography>
          </Box>
        )}

        {/* Qwen multimodal description (smart mode) */}
        {window.multimodal_description && (
          <Box sx={{ mb: 1.5, p: 1.5, backgroundColor: 'grey.50', borderRadius: 1, border: '1px solid', borderColor: 'grey.200' }}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
              Análisis Qwen (qwen3-vl:8b)
            </Typography>
            <Typography variant="body2">
              {window.multimodal_description}
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Emoción dominante
            </Typography>
            <Typography variant="body2" fontWeight={600}>
              {translateLabel(window.emotion_primary ?? window.dominant_emotion, EMOTION_LABELS)}
              {window.emotion_confidence != null && (
                <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>
                  ({(window.emotion_confidence * 100).toFixed(0)}%)
                </Typography>
              )}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Contacto visual
            </Typography>
            <Typography variant="body2" fontWeight={600}>
              {window.is_looking_at_camera ? 'Sí' : 'No'}
            </Typography>
          </Box>
          {window.attention_direction && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Dirección de atención
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {translateLabel(window.attention_direction, ATTENTION_LABELS)}
              </Typography>
            </Box>
          )}
          {window.engagement_level && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Nivel de engagement
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {translateLabel(window.engagement_level, ENGAGEMENT_LABELS)}
              </Typography>
            </Box>
          )}
          {window.moment_type && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Tipo de momento
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {translateLabel(window.moment_type, MOMENT_TYPE_LABELS)}
              </Typography>
            </Box>
          )}
          {!window.is_smart_window && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Fotogramas analizados
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {window.frame_count}
              </Typography>
            </Box>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
}

export default function VisionLane({
  personId,
  windows,
  durationSeconds,
  pixelsPerSecond,
  laneHeight,
}: VisionLaneProps): JSX.Element {
  const lh = laneHeight ?? 28;
  const width = Math.max(1, durationSeconds * pixelsPerSecond);

  const [selectedWindow, setSelectedWindow] = useState<VisionWindow | null>(null);

  const filtered = windows.filter((w) => w.person_id === personId);

  return (
    <>
      <svg width={width} height={lh} style={{ display: 'block' }}>
        {filtered.map((window) => {
          const wx = window.windowStart * pixelsPerSecond;
          const ww = Math.max(2, (window.windowEnd - window.windowStart) * pixelsPerSecond);
          const eyeX = wx + ww - 5;
          const eyeY = 5;

          const hasImage = Boolean(window.image_url || window.frame_url);

          return (
            <Tooltip
              key={`${window.person_id}-${window.windowStart}`}
              title={`${personId} | ${window.dominant_emotion} | ${window.is_looking_at_camera ? 'Mirando cámara' : 'No mira cámara'} | ${formatTime(window.windowStart)}→${formatTime(window.windowEnd)}${hasImage ? ' | Clic para ver imagen' : ''}`}
            >
              <g
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedWindow(window)}
                role="button"
                aria-label={`${personId}: ${window.dominant_emotion}, ${window.is_looking_at_camera ? 'mirando' : 'no mirando'}, ${formatTime(window.windowStart)}-${formatTime(window.windowEnd)}`}
              >
                <rect
                  x={wx}
                  y={0}
                  width={ww}
                  height={lh}
                  fill={getEmotionColor(window.dominant_emotion)}
                  opacity={0.8}
                />
                {window.is_looking_at_camera && (
                  <circle cx={eyeX} cy={eyeY} r={3} fill="white" opacity={0.9} />
                )}
                {/* Small camera icon indicator when a thumbnail is available */}
                {hasImage && ww >= 10 && (
                  <rect
                    x={wx + 2}
                    y={lh - 6}
                    width={5}
                    height={4}
                    rx={1}
                    fill="white"
                    opacity={0.75}
                  />
                )}
              </g>
            </Tooltip>
          );
        })}
      </svg>

      {selectedWindow !== null && (
        <FrameModal
          window={selectedWindow}
          personId={personId}
          onClose={() => setSelectedWindow(null)}
        />
      )}
    </>
  );
}
