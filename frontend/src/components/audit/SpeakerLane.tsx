import { Tooltip } from '@mui/material';
import { SpeakerTurn, Interruption } from '../../types/audit';
import { formatTime } from './utils/timeFormat';
import InterruptionMarker from './InterruptionMarker';

interface SpeakerLaneProps {
  speakerId: string;
  turns: SpeakerTurn[];
  interruptions: Interruption[];
  durationSeconds: number;
  pixelsPerSecond: number;
  color: string;
  laneHeight?: number;
}

export default function SpeakerLane({
  speakerId,
  turns,
  interruptions,
  durationSeconds,
  pixelsPerSecond,
  color,
  laneHeight,
}: SpeakerLaneProps): JSX.Element {
  const lh = laneHeight ?? 36;
  const width = Math.max(1, durationSeconds * pixelsPerSecond);

  const speakerInterruptions = interruptions.filter(
    (interruption) => interruption.interrupter_id === speakerId,
  );

  return (
    <svg width={width} height={lh}>
      <rect width="100%" height="100%" fill="#f5f5f5" />
      {turns.map((turn) => (
        <Tooltip
          key={turn.start}
          title={`${speakerId}: ${formatTime(turn.start)} → ${formatTime(turn.end)} (${turn.duration.toFixed(1)}s)`}
        >
          <rect
            x={turn.start * pixelsPerSecond}
            width={Math.max(2, turn.duration * pixelsPerSecond)}
            y={2}
            height={lh - 4}
            fill={color}
            opacity={0.7}
            rx={3}
            aria-label={`Speaker ${speakerId}: ${turn.start}s – ${turn.end}s`}
            style={{ cursor: 'pointer' }}
          />
        </Tooltip>
      ))}
      {speakerInterruptions.map((interruption) => (
        <InterruptionMarker
          key={`int-${interruption.time}`}
          interruption={interruption}
          pixelsPerSecond={pixelsPerSecond}
          y={0}
          height={lh}
        />
      ))}
    </svg>
  );
}
