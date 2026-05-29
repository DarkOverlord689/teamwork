import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar } from "react-chartjs-2";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface RubricScores {
  contributes_to_team_meetings: number;
  facilitates_contributions: number;
  fosters_constructive_climate: number;
  responds_to_conflict: number;
  individual_contributions_outside: number;
}

interface RubricRadarChartProps {
  scores: RubricScores;
  label: string;
}

const DIMENSION_LABELS = [
  "Contribuye en Reuniones",
  "Facilita Contribuciones",
  "Clima Constructivo",
  "Responde a Conflictos",
  "Contrib. Fuera de Reuniones",
];

export default function RubricRadarChart({ scores, label }: RubricRadarChartProps) {
  const dataValues = [
    scores.contributes_to_team_meetings,
    scores.facilitates_contributions,
    scores.fosters_constructive_climate,
    scores.responds_to_conflict,
    scores.individual_contributions_outside,
  ];

  const chartData = {
    labels: DIMENSION_LABELS,
    datasets: [
      {
        label,
        data: dataValues,
        backgroundColor: "rgba(25, 118, 210, 0.2)",
        borderColor: "rgba(25, 118, 210, 1)",
        borderWidth: 2,
        pointBackgroundColor: "rgba(25, 118, 210, 1)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(25, 118, 210, 1)",
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      r: {
        min: 0,
        max: 20,
        ticks: {
          stepSize: 2,
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: "top" as const,
      },
    },
  };

  return <Radar data={chartData} options={options} />;
}
