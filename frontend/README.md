# SMATC-UPAO Frontend

Aplicación React para dashboard docente de análisis de trabajo colaborativo.

## Requisitos

- Node.js 18+
- npm o yarn

## Instalación

```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar desarrollo
npm run dev
```

## Estructura del Proyecto

```
frontend/
├── src/
│   ├── components/       # Componentes React
│   │   ├── common/       # Componentes compartidos
│   │   ├── dashboard/    # Dashboard y métricas
│   │   ├── upload/       # Carga de videos
│   │   ├── reports/      # Visualización de reportes
│   │   └── layout/       # Layout y navegación
│   ├── hooks/           # Custom hooks
│   ├── services/        # Servicios API
│   ├── store/           # Redux store
│   │   ├── slices/      # Redux slices
│   │   └── index.ts
│   ├── pages/           # Páginas
│   ├── types/           # Tipos TypeScript
│   ├── utils/           # Utilidades
│   ├── styles/          # Estilos globales
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Scripts

- `npm run dev` - Iniciar servidor de desarrollo
- `npm run build` - Build de producción
- `npm run lint` - Verificar código
- `npm run preview` - Preview de build

## Dependencias

- React 18+
- TypeScript 5.3+
- Material UI 5.15+
- Redux Toolkit
- React Router 6
- Chart.js 4.4+
- D3.js 7.8+
- Video.js 8.6+