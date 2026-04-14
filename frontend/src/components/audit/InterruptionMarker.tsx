import { Tooltip } from '@mui/material';
import { Interruption } from '../../types/audit';
import { formatMs } from './utils/timeFormat';

interface InterruptionMarkerProps {
  interruption: Interruption;
  pixelsPerSecond: number;
  y: number;
  height: number;
}

export default function InterruptionMarker({
  interruption,
  pixelsPerSecond,
  y,
  height,
}: InterruptionMarkerProps): JSX.Element {
  const x = interruption.time * pixelsPerSecond;
  const cy = y + height / 2;
  const half = height / 4;

  const diamondPath = `M ${x} ${cy - half} L ${x + half} ${cy} L ${x} ${cy + half} L ${x - half} ${cy} Z`;

  const tooltipTitle = `Interruptor: ${interruption.interrupter_id} | Interrumpido: ${interruption.interrupted_id} | Duración: ${formatMs(interruption.overlap_duration * 1000)} | Tipo: ${interruption.interruption_type}`;

  return (
    <Tooltip title={tooltipTitle}>
      <g style={{ cursor: 'pointer' }}>
        <path d={diamondPath} fill="#ED6C02" />
      </g>
    </Tooltip>
  );
}
