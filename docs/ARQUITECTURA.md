# SMATC-UPAO - Documentación del Proyecto

## Arquitectura del Sistema

### Capas del Sistema

1. **Capa de Presentación** (Frontend)
   - React + TypeScript
   - Material UI para componentes
   - Redux Toolkit para gestión de estado
   - Chart.js y D3.js para visualizaciones

2. **Capa de Aplicación** (Backend API)
   - FastAPI (Python 3.11+)
   - Autenticación JWT
   - Orquestación de pipelines

3. **Capa de Procesamiento**
   - **Visión por computador**: MediaPipe, Qwen2-VL, OpenCV
   - **Audio**: Pyannote, Whisper, Librosa
   - **Fusión + LLM**: Llama 3.1 (8B), RAG

4. **Capa de Datos**
   - PostgreSQL (metadatos y métricas)
   - Redis (colas y cache)
   - MinIO (almacenamiento de videos)

## Módulos de Procesamiento

### Procesamiento Visual
- Detección facial con MediaPipe Face Mesh
- Análisis de contacto visual (Qwen2-VL)
- Reconocimiento de gestos (asentimiento, ceño)
- Estimación de pose corporal

### Procesamiento Auditivo
- Diarización de hablantes (Pyannote 3.1)
- Transcripción automática (Whisper large-v3)
- Detección de interrupciones
- Análisis de turnos de habla

### Fusión Multimodal
- Alineación temporal video-audio
- Cálculo de métricas (CV de participación, % contacto visual)
- Generación de explicaciones (LLM + RAG)
- Mapeo a rúbrica UPAO

## Endpoints de API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | Autenticación |
| POST | `/api/v1/upload/` | Subir video |
| GET | `/api/v1/analysis/groups` | Listar grupos |
| GET | `/api/v1/analysis/groups/{id}` | Detalles del grupo |
| GET | `/api/v1/reports/{group_id}` | Obtener reportes |
| POST | `/api/v1/validate/{id}` | Validar análisis |

## Despliegue

El proyecto usa Docker Compose para orquestación:

```bash
cd infrastructure
docker-compose up -d
```

Servicios:
- `nginx`: Proxy reverso
- `frontend`: Aplicación React
- `backend`: API FastAPI
- `celery-worker`: Procesamiento asíncrono
- `postgres`: Base de datos
- `redis`: Colas y cache
- `minio`: Almacenamiento S3

## Variables de Entorno

Ver `.env.example` en el directorio `backend/`.