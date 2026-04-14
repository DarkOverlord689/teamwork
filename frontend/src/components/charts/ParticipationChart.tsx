import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface StudentEntry {
  name: string;
  speaking_time: number;
}

interface ParticipationChartProps {
  students: StudentEntry[];
}

const STUDENT_COLORS = [
  "#1976d2",
  "#4caf50",
  "#ff9800",
  "#9c27b0",
  "#f44336",
  "#00bcd4",
  "#795548",
  "#607d8b",
];

export default function ParticipationChart({ students }: ParticipationChartProps) {
  const labels = students.map((s) => s.name);
  const data = students.map((s) => s.speaking_time);
  const backgroundColors = students.map((_, i) => STUDENT_COLORS[i % STUDENT_COLORS.length]);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Tiempo de Habla (segundos)",
        data,
        backgroundColor: backgroundColors,
        borderColor: backgroundColors.map((c) => c),
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Tiempo de Habla por Estudiante",
      },
      tooltip: {
        callbacks: {
          label: (ctx: { raw: unknown }) => `${ctx.raw} segundos`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Segundos",
        },
      },
    },
  };

  return <Bar data={chartData} options={options} />;
}
