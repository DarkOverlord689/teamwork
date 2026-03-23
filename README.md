🚀 SMATC-UPAO
Sistema Multimodal de Análisis de Trabajo Colaborativo

Plataforma web para el análisis de grabaciones de trabajo grupal utilizando procesamiento multimodal (visual, auditivo y textual) junto con modelos LLMs para generar métricas objetivas de participación y reportes explicativos.

🧠 Descripción

SMATC-UPAO permite evaluar dinámicas de trabajo colaborativo mediante el análisis automático de video y audio, generando insights cuantitativos y cualitativos alineados a rúbricas académicas.

🛠️ Tech Stack
🔙 Backend
Python 3.11+
FastAPI
PyTorch
MediaPipe
Pyannote
Whisper
Llama 3.1
🎨 Frontend
React 18+
TypeScript 5.3+
Material UI
Redux Toolkit
D3.js
🗄️ Base de Datos
PostgreSQL 16
Redis 7.2
MinIO (compatible con S3)
⚙️ Infraestructura
Docker
Docker Compose
Nginx
📁 Estructura del Proyecto
cogno/
├── backend/          # API REST con FastAPI
├── frontend/         # Aplicación React
├── infrastructure/   # Docker y configuración de despliegue
└── docs/             # Documentación
⚡ Quick Start
🐳 Con Docker (recomendado)
# Clonar el repositorio
git clone <repo-url>
cd teamwork

# Iniciar servicios
docker-compose up -d
💻 Desarrollo local

Consulta los README específicos de cada módulo:

backend/README.md
frontend/README.md
🧩 Módulos del Sistema
👁️ 1. Procesamiento Visual
Detección facial (MediaPipe Face Mesh)
Análisis de contacto visual (Qwen2-VL)
Reconocimiento de gestos (OpenCV)
Estimación de pose (MediaPipe)
🔊 2. Procesamiento Auditivo
Diarización de hablantes (Pyannote 3.1)
Transcripción automática (Whisper large-v3)
Detección de interrupciones
Análisis de turnos de habla
🔗 3. Fusión Multimodal
Alineación temporal video-audio
Cálculo de métricas:
Coeficiente de variación de participación
% de contacto visual
Generación de explicaciones (Llama 3.1 + RAG)
Mapeo a rúbrica UPAO
📊 4. Interfaz Docente
Dashboard de métricas
Carga de videos
Visualización de reportes
Validación de resultados
🌐 API Endpoints
Endpoint	Método	Descripción
/api/v1/upload	POST	Subir video
/api/v1/analysis/{group_id}	GET	Obtener análisis
/api/v1/reports/{group_id}	GET	Descargar reporte
/api/v1/metrics/{student_id}	GET	Métricas por estudiante
/api/v1/validate/{analysis_id}	POST	Validar / corregir análisis
📚 Documentación
📐 Diseño del sistema
 – Arquitectura completa
🔙 Backend
🎨 Frontend
⚠️ Requisitos
Python 3.11+
Node.js 18+
Docker 24+
Hardware recomendado:
32 GB RAM
GPU NVIDIA (GTX 1660 o superior)
🔒 Licencia

Proyecto privado – Universidad Privada Antenor Orrego (UPAO)
