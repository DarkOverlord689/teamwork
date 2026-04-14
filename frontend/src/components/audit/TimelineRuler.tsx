import { tickInterval, formatTime } from './utils/timeFormat';

interface TimelineRulerProps {
  durationSeconds: number;
  pixelsPerSecond: number;
  height?: number;
}

export default function TimelineRuler({
  durationSeconds,
  pixelsPerSecond,
  height,
}: TimelineRulerProps): JSX.Element {
  const h = height ?? 32;

  if (durationSeconds <= 0) {
    return <svg width={0} height={h} />;
  }

  const width = durationSeconds * pixelsPerSecond;
  const interval = tickInterval(durationSeconds);
  const pps = pixelsPerSecond;

  const ticks: number[] = [];
  for (let t = 0; t <= durationSeconds; t += interval) {
    ticks.push(t);
  }

  return (
    <svg width={width} height={h}>
      <line x1={0} y1={h - 1} x2={width} y2={h - 1} stroke="#bdbdbd" />
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={t * pps}
            y1={h - 8}
            x2={t * pps}
            y2={h - 1}
            stroke="#757575"
          />
          <text x={t * pps + 2} y={h - 12} fontSize={10} fill="#616161">
            {formatTime(t)}
          </text>
        </g>
      ))}
    </svg>
  );
}
