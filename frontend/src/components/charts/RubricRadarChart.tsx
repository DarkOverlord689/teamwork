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
  collaboration: number;
  communication: number;
  responsibility: number;
  leadership: number;
  technical_contribution: number;
}

interface RubricRadarChartProps {
  scores: RubricScores;
  label: string;
}

const DIMENSION_LABELS = [
  "Colaboración",
  "Comunicación",
  "Responsabilidad",
  "Liderazgo",
  "Contribución Técnica",
];

export default function RubricRadarChart({ scores, label }: RubricRadarChartProps) {
  const dataValues = [
    scores.collaboration,
    scores.communication,
    scores.responsibility,
    scores.leadership,
    scores.technical_contribution,
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
