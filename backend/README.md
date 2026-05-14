# SMATC-UPAO Backend

API REST desarrollada con FastAPI para el procesamiento multimodal de videos de trabajo colaborativo.

## Requisitos

- Python 3.11+
- PostgreSQL 16
- Redis 7.2
- MinIO (opcional, para almacenamiento de videos)

## Instalación

```bash
cd backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Copiar configuración
cp .env.example .env

# Ejecutar migraciones
alembic upgrade head

# Iniciar servidor
uvicorn app.main:app --reload
```

## Estructura del Proyecto

```
backend/
├── app/
│   ├── api/              # Endpoints y routers
│   │   ├── v1/
│   │   │   ├── auth.py
│   │   │   ├── upload.py
│   │   │   ├── analysis.py
│   │   │   ├── reports.py
│   │   │   └── validate.py
│   │   └── deps.py
│   ├── core/             # Módulos de procesamiento
│   │   ├── vision/       # Procesamiento visual
│   │   │   ├── face_detector.py
│   │   │   ├── gaze_tracker.py
│   │   │   ├── gesture_recognizer.py
│   │   │   └── pose_estimator.py
│   │   ├── audio/        # Procesamiento auditivo
│   │   │   ├── diarizer.py
│   │   │   ├── transcriber.py
│   │   │   ├── interruption_detector.py
│   │   │   └── turn_analyzer.py
│   │   └── fusion/       # Fusión multimodal
│   │       ├── alignment.py
│   │       ├── metrics_calculator.py
│   │       ├── explanation_generator.py
│   │       └── rubric_mapper.py
│   ├── models/           # Modelos SQLAlchemy
│   │   ├── group.py
│   │   ├── student.py
│   │   ├── session.py
│   │   └── metrics.py
│   ├── schemas/          # Schemas Pydantic
│   │   ├── group.py
│   │   ├── analysis.py
│   │   └── report.py
│   ├── services/         # Lógica de negocio
│   │   ├── analysis_service.py
│   │   ├── report_service.py
│   │   └── storage_service.py
│   ├── tasks/           # Tareas Celery
│   │   └── processing.py
│   ├── utils/           # Utilidades
│   │   ├── config.py
│   │   ├── security.py
│   │   └── storage.py
│   ├── database.py
│   └── main.py
├── alembic/             # Migraciones
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Variables de Entorno

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/smatc

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO/S3
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=videos

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Models
LLAMA_MODEL_PATH=/models/llama-3.1-8b
WHISPER_MODEL=large-v3

# Processing
MAX_WORKERS=4
VIDEO_FPS=5
```

## API Documentation

Una vez iniciado el servidor, accede a:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Procesamiento asíncrono

El sistema utiliza Celery para procesamiento asíncrono de videos:

```bash
# Iniciar worker de Celery
celery -A app.tasks.celery_app worker --loglevel=info
```

## Tests

```bash
pytest tests/ -v --cov
```

## Licencia

Privado - UPAO