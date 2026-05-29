# Sistema de Evaluación Automática de Trabajo Colaborativo mediante Rúbrica VALUE (AAC&U)

## Documento Técnico — SMATC-UPAO v2.1

### Estructurado por Objetivos Específicos de Tesis

---

## Contenido

1. [Resumen](#resumen)
2. [OE1: Parametrización de indicadores VALUE en biomarcadores cuantificables](#oe1-parametrización-de-indicadores-value-en-biomarcadores-cuantificables)
3. [OE2: Diseño de la arquitectura lógico-física del sistema](#oe2-diseño-de-la-arquitectura-lógico-física-del-sistema)
4. [OE3: Construcción del prototipo del sistema cognitivo multimodal](#oe3-construcción-del-prototipo-del-sistema-cognitivo-multimodal)
5. [OE4: Diseño experimental para evaluación estadística de precisión y concordancia](#oe4-diseño-experimental-para-evaluación-estadística-de-precisión-y-concordancia)
6. [OE5: Instrumento de aceptabilidad y usabilidad (TAM)](#oe5-instrumento-de-aceptabilidad-y-usabilidad-tam)
7. [Limitaciones del sistema](#limitaciones-del-sistema)
8. [Trabajo futuro](#trabajo-futuro)
9. [Referencias](#referencias)

---

## Resumen

El presente documento describe el sistema de evaluación automática del trabajo colaborativo implementado en la plataforma SMATC-UPAO, el cual utiliza análisis multimodal (video + audio + NLP) para asignar puntuaciones alineadas con la rúbrica VALUE (*Valid Assessment of Learning in Undergraduate Education*) de *Teamwork* desarrollada por la AAC&U. El contenido se organiza según los objetivos específicos de la tesis, abarcando desde la parametrización de indicadores colaborativos en biomarcadores cuantificables (OE1), el diseño arquitectónico del sistema (OE2), la construcción del prototipo (OE3), el diseño experimental para la validación estadística con ICC (OE4), y el instrumento de aceptabilidad docente TAM (OE5). Se incluyen fórmulas matemáticas completas, diagramas de arquitectura, estructuras de datos y el plan experimental para la fase de evaluación.

**Palabras clave:** trabajo colaborativo, rúbrica VALUE, AAC&U, análisis multimodal, ICC, evaluación automática, fusión audiovisual, diarización, reconocimiento de emociones.

---

## OE1: Parametrización de indicadores VALUE en biomarcadores cuantificables

### 1.1 Contexto del objetivo

OE1 establece: *"Analizar los requerimientos técnicos y operacionales mediante la parametrización de indicadores de trabajo colaborativo de la rúbrica VALUE en biomarcadores cuantificables por inteligencia artificial (contacto visual, turnos de habla, gestos de asentimiento, duración de intervenciones, solapamientos)."*

Esta sección presenta el mapeo sistemático entre cada dimensión de la rúbrica VALUE y los biomarcadores extraídos automáticamente de las señales de audio y video.

### 1.2 La rúbrica VALUE de Teamwork (AAC&U)

La rúbrica VALUE para *Teamwork* define cinco dimensiones del trabajo colaborativo, cada una con cuatro niveles progresivos:

| Nivel VALUE | Puntaje equivalente (0-20) | Descripción |
|---|---|---|
| **Benchmark (1)** | 5.0 | Desempeño básico |
| **Milestone 2 (2)** | 10.0 | Desempeño intermedio |
| **Milestone 3 (3)** | 15.0 | Buen desempeño |
| **Capstone (4)** | 20.0 | Desempeño ejemplar |

### 1.3 Mapeo sistemático: Indicador VALUE → Biomarcador → Tecnología

| Dimensión VALUE | Indicador cualitativo | Biomarcador cuantificable | Tecnología extractora | Unidad | Fuente |
|---|---|---|---|---|---|
| **D1: Contributes to Team Meetings** | Participación activa | `participation_ratio` | Deepgram Nova-2 + TurnAnalyzer | ratio [0,1] | Audio |
| | Frecuencia de intervenciones | `turn_count` | Deepgram Nova-2 + TurnAnalyzer | conteo | Audio |
| | Iniciativa en temas nuevos | `initiation_count` | TurnAnalyzer (gap > 2s entre turnos) | conteo | Audio |
| | Atención cuando otros hablan | `gaze_contact_percentage` | MediaPipe Face Mesh + GazeEstimator | porcentaje | Video |
| **D2: Facilitates Contributions** | Escucha activa | `back_channel_count` | TurnAnalyzer (turnos < 0.5s) | conteo | Audio |
| | Atención visual hacia compañeros | `gaze_contact_percentage` | MediaPipe Face Mesh + GazeEstimator | porcentaje | Video |
| | Interrupciones cooperativas | `interruption_count` (tipo COOPERATIVE) | InterruptionDetector + NLP | conteo | Audio |
| **D3: Individual Contributions Outside** | (No observable en sesión grabada) | — | Requiere datos externos (LMS, repos) | — | Externo |
| **D4: Fosters Constructive Climate** | Tono emocional positivo | `dominant_emotion` | CLIP ViT-B/32 (zero-shot) | categórico (9 clases) | Video |
| | Lenguaje corporal de respeto | `avg_body_orientation` | MediaPipe Pose | grados [0,90] | Video |
| | Gestos de asentimiento | `gesture_type` (nod) | GestureAnalyzer (ventana landmarks) | categórico | Video |
| | Respeto por turnos ajenos | `interruption_count` (tipo DISRUPTIVE) | InterruptionDetector | conteo | Audio |
| **D5: Responds to Conflict** | Dinámica grupal saludable | `disruptive_interruption_rate` | ParticipationAggregator | ratio [0,1] | Audio |
| | Fluidez en transiciones | `turn_synchronization_score` | TurnAnalyzer | ratio [0,1] | Audio |
| | Balance interrupción/ceder turno | `interruption_count` / `interrupted_count` | InterruptionDetector | conteo | Audio |

### 1.4 Métricas base extraídas del pipeline

#### 1.4.1 Métricas por estudiante (`StudentMetrics`)

| Métrica | Tipo | Descripción |
|---|---|---|
| `student_id` | string | Identificador mapeado de `speaker_id` o `person_id` |
| `speaking_time_seconds` | float | Tiempo total de habla |
| `turn_count` | int | Cantidad de turnos de habla |
| `interruption_count` | int | Veces que interrumpió a otro |
| `interrupted_count` | int | Veces que fue interrumpido |
| `participation_ratio` | float | Fracción del tiempo total de habla del grupo |
| `gaze_contact_percentage` | float | % de ventanas mirando a cámara/compañeros |
| `avg_body_orientation` | float | Orientación corporal promedio (0° = frente) |
| `dominant_emotion` | string | Emoción más frecuente (9 clases) |
| `initiation_count` | int | Turnos que inician tras silencio >2s o cambio de hablante |
| `back_channel_count` | int | Turnos de afirmación/escucha (< 0.5s) |

#### 1.4.2 Métricas grupales (`GroupMetrics`)

| Métrica | Tipo | Descripción |
|---|---|---|
| `total_students` | int | Número de estudiantes identificados |
| `participation_cv` | float | CV de tiempos de habla (σ/μ). CV < 0.3 → equitativo |
| `disruptive_interruption_rate` | float | Interrupciones disruptivas / total de turnos |
| `turn_synchronization_score` | float | Transiciones suaves (gap 0-1s) / total. Rango [0,1] |
| `avg_gaze_contact_percentage` | float | Media del contacto visual de todos los estudiantes |
| `silence_ratio` | float | Fracción de la sesión sin habla |

### 1.5 Metodología de puntuación

Cada dimensión VALUE se evalúa como una **suma ponderada** de sub-métricas normalizadas al rango [0, 1], multiplicada por 20:

$$S_d = 20 \cdot \sum_{i=1}^{n} w_i \cdot m_i$$

Donde $S_d$ es el puntaje de la dimensión $d$, $w_i$ son los pesos (suman 1.0), y $m_i$ son las sub-métricas normalizadas.

El puntaje grupal es el promedio aritmético de los puntajes individuales:

$$S_d^{grupo} = \frac{1}{N} \sum_{j=1}^{N} S_d^{(j)}$$

Todos los puntajes se recortan al rango [0, 20] mediante $\text{clamp}(x) = \max(0, \min(20, x))$.

### 1.6 Dimensión 1: Contributes to Team Meetings

**Definición VALUE:** Participación activa en reuniones, compartir ideas, ofrecer sugerencias, ayudar a que el equipo avance.

| Sub-métrica | Peso | Fórmula |
|---|---|---|
| **Participation engagement** | 0.40 | Función escalonada del ratio participación/porción equitativa |
| **Idea contribution** | 0.35 | $\min(\frac{\text{initiation\_count}}{\text{turn\_count} \cdot 0.5},\ 1.0)$ |
| **Turn frequency** | 0.25 | $\frac{\min(\frac{\text{turn\_count}}{\text{expected\_turns}},\ 1.5)}{1.5}$ |

**Participation engagement** ($m_1$). Sea $r = \frac{\text{participation\_ratio}}{\text{equal\_share}}$ con $\text{equal\_share} = \frac{1}{N}$:

$$m_1 = \begin{cases}
0.6 & \text{si } r > 2.0 \quad \text{(domina la conversación)} \\
1.0 - (r - 1.0) \cdot 0.2 & \text{si } 1.0 < r \leq 2.0 \\
0.5 + (r - 0.5) \cdot 0.5 & \text{si } 0.5 < r \leq 1.0 \\
r & \text{si } r \leq 0.5 \quad \text{(participación escasa)}
\end{cases}$$

**Puntaje final:**

$$S_1 = 20 \cdot (0.40 \cdot m_1 + 0.35 \cdot m_2 + 0.25 \cdot m_3)$$

### 1.7 Dimensión 2: Facilitates Contributions of Team Members

**Definición VALUE:** Involucrar a otros, construir sobre ideas ajenas, preguntar, facilitar contribuciones.

| Sub-métrica | Peso | Fórmula |
|---|---|---|
| **Active listening** | 0.35 | $0.6 \cdot \text{bc\_score} + 0.4 \cdot \text{gaze\_fraction}$ |
| **Cooperative engagement** | 0.40 | $0.4 \cdot \text{gaze\_fraction} + 0.6 \cdot \text{cooperation}$ |
| **Engagement signals** | 0.25 | $\text{gaze\_fraction} = \frac{\text{gaze\_contact\_percentage}}{100}$ |

Donde:

$$\text{bc\_score} = \min\left(\frac{\text{back\_channel\_count}}{\text{turn\_count} \cdot 0.5},\ 1.0\right)$$

$$\text{cooperation} = \max\left(0,\ 1.0 - \frac{\text{interruption\_count}}{\max(\text{turn\_count},\ 1)}\right)$$

**Puntaje final:**

$$S_2 = 20 \cdot (0.35 \cdot m_1 + 0.40 \cdot m_2 + 0.25 \cdot m_3)$$

### 1.8 Dimensión 3: Individual Contributions Outside Team Meetings

**Definición VALUE:** Tareas completadas, investigación independiente, materiales preparados.

**Limitación:** Esta dimensión no puede evaluarse a partir de la grabación de video/audio. Requiere integración con fuentes externas (LMS, repositorios Git, plataformas de entrega). El sistema asigna **0.0** como placeholder.

### 1.9 Dimensión 4: Fosters Constructive Team Climate

**Definición VALUE:** Clima positivo, trato respetuoso, tono positivo, lenguaje corporal de apoyo.

| Sub-métrica | Peso | Descripción |
|---|---|---|
| **Emotional tone** | 0.35 | Puntaje según emoción facial dominante (tabla abajo) |
| **Respect signals** | 0.35 | $0.5 \cdot \text{respect} + 0.5 \cdot \text{gaze\_fraction}$ |
| **Body language** | 0.30 | $\max(0,\ 1.0 - \frac{\text{avg\_body\_orientation}}{90°})$ |

**Emotional tone** ($m_1$) — puntajes predefinidos por emoción detectada por CLIP:

| Emoción | Puntaje |
|---|---|
| `attentive`, `happy` | 0.9 |
| `surprised`, `neutral` | 0.7 |
| `sad`, `fearful` | 0.35 |
| `angry`, `disgusted`, `contemptuous` | 0.1 |
| Desconocida | 0.5 |

$$\text{respect} = \max\left(0,\ 1.0 - 1.5 \cdot \frac{\text{interruption\_count}}{\max(\text{turn\_count},\ 1)}\right)$$

**Puntaje final:**

$$S_4 = 20 \cdot (0.35 \cdot m_1 + 0.35 \cdot m_2 + 0.30 \cdot m_3)$$

### 1.10 Dimensión 5: Responds to Conflict

**Definición VALUE:** Abordar desacuerdos constructivamente, negociar, resolver conflictos.

| Sub-métrica | Peso | Fórmula |
|---|---|---|
| **Conflict management** (grupal) | 0.50 | $0.5 \cdot \text{disruptive\_score} + 0.5 \cdot \text{turn\_sync}$ |
| **Constructive engagement** (individual) | 0.50 | $1.0 - \left|\frac{\text{interruption\_count}}{\text{total\_intr}} - 0.5\right| \cdot 2.0$ |

Donde:

$$\text{disruptive\_score} = \max(0,\ 1.0 - 2.0 \cdot \text{disruptive\_interruption\_rate})$$

$$\text{total\_intr} = \text{interruption\_count} + \text{interrupted\_count}$$

**Puntaje final:**

$$S_5 = 20 \cdot (0.50 \cdot m_1 + 0.50 \cdot m_2)$$

### 1.11 Pesos configurables de la rúbrica

```python
RUBRIC_WEIGHTS = {
    "contributes_to_team_meetings": {
        "participation_engagement": 0.40,
        "idea_contribution":      0.35,
        "turn_frequency":         0.25,
    },
    "facilitates_contributions": {
        "active_listening":       0.35,
        "cooperative_engagement": 0.40,
        "engagement_signals":     0.25,
    },
    "fosters_constructive_climate": {
        "emotional_tone":         0.35,
        "respect_signals":        0.35,
        "body_language":          0.30,
    },
    "responds_to_conflict": {
        "conflict_management":      0.50,
        "constructive_engagement":  0.50,
    },
}
```

---

## OE2: Diseño de la arquitectura lógico-física del sistema

### 2.1 Contexto del objetivo

OE2 establece: *"Diseñar la arquitectura lógico-física del sistema, estableciendo las canalizaciones de datos para los modelos de Visión Computacional, Análisis de Voz y NLP."*

### 2.2 Arquitectura general

```
┌──────────────────────────────────────────────────────────────────┐
│                        FASTAPI (REST)                            │
│  POST /api/v1/upload/   │   GET /fusion/sessions/{id}/result    │
│  GET /validate/{id}     │   GET /teacher/dashboard              │
└─────────────┬────────────────────────────────────────────────────┘
              │                        ▲
              ▼                        │
┌─────────────────────────┐   ┌───────┴──────────┐
│      CELERY + REDIS     │   │   POSTGRESQL     │
│  (Task Queue + Backend) │   │  AnalysisSession │
│                         │   │  RubricScore     │
│  ┌───────────────────┐  │   │  result_data     │
│  │ process_video_task│  │   │  (JSONB)         │
│  └────────┬──────────┘  │   └──────────────────┘
│           │ chain       │            ▲
│  ┌────────▼──────────┐  │            │
│  │ process_audio_task│  │   ┌────────┴──────────┐
│  └────────┬──────────┘  │   │     MINIO S3      │
│           │ chain       │   │  (video storage)  │
│  ┌────────▼──────────┐  │   └───────────────────┘
│  │ process_fusion    │──┼───→ persist JSONB
│  │      _task        │  │
│  └───────────────────┘  │
└─────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌──────────┐
│ VISION │ │ AUDIO  │ │ FUSION   │
│PIPELINE│ │PIPELINE│ │PIPELINE  │
└───┬────┘ └───┬────┘ └────┬─────┘
    │          │            │
    │  ┌───────┴───────┐    │
    │  │   APIs CLOUD  │    │
    │  │  Deepgram     │    │
    │  │  OpenAI       │    │
    │  └───────────────┘    │
    │                       │
    ▼                       ▼
┌─────────────────────────────────────┐
│         MODELOS LOCALES             │
│  MediaPipe Face Mesh / Pose        │
│  CLIP ViT-B/32 (emotion)           │
│  InceptionResnetV1 (face tracking)  │
│  librosa (audio extraction)         │
└─────────────────────────────────────┘
```

### 2.3 Pipeline de Visión (Módulo 2.1)

| Componente | Tecnología | Entrada | Salida |
|---|---|---|---|
| FrameExtractor | OpenCV 4.x | Video MP4 | Frames BGR (3 FPS, máx. 500) |
| FaceDetector | MediaPipe Face Mesh | Frame | 468 landmarks 3D por rostro |
| PersonTracker | InceptionResnetV1 (VGGFace2) | Landmarks | `person_id` estable (embeddings 512-d + IoU) |
| EmotionClassifier | CLIP ViT-B/32 (zero-shot) | Rostro recortado | 9 emociones con confianza |
| GazeEstimator | Geométrico (landmarks iris 468-477) | Landmarks | yaw/pitch → CAMERA/SCREEN/PEER/AWAY |
| GestureAnalyzer | Ventana deslizante landmarks | Secuencia landmarks | nod/shake/frown |
| PoseEstimator | MediaPipe Pose | Frame | Ángulo de hombros (0°-90°) |
| SpeakerFaceMapper | Votación co-ocurrencia + cosine sim | speaker_id + person_id | Mapeo speaker↔person |

**Parámetros de configuración:**
- FPS: 3.0 (configurable `vision_fps`)
- Máximo de frames: 500 (configurable `vision_max_frames`)
- Umbral de similitud facial: 0.6 (cosine)
- Ventana de gestos: 5 frames

### 2.4 Pipeline de Audio (Módulo 2.2)

| Componente | Tecnología | Entrada | Salida |
|---|---|---|---|
| AudioExtractor | librosa 0.10.x | Video MP4 | Waveform mono float32 16kHz |
| Diarizer | Deepgram Nova-2 API | Waveform | `SpeakerSegment[]` (inicio, fin, speaker_id) |
| Transcriber | Deepgram Nova-2 API | Waveform | `TranscriptSegment[]` con word-level timestamps |
| TurnAnalyzer | Algorítmico | Segments | `SpeakerTurn[]` (merge gap < 0.5s), overlaps, CV |
| InterruptionDetector | Algorítmico + NLP | Turns + Transcript | DISRUPTIVE / COOPERATIVE / BACK_CHANNEL |
| ParticipationAggregator | Algorítmico | Turns + Interruptions | `SpeakerMetrics[]` + `AudioSessionMetrics` |
| KeyFrameSelector | Algorítmico | Transcript + Turns | `KeyMoment[]` (speech onset, preguntas, hesitaciones) |

**Parámetros de configuración:**
- Sample rate: 16000 Hz
- Gap de merge de turnos: 0.5s
- Duración mínima de segmento: 0.5s
- Ventana de interrupción disruptiva: 0.2s restante del turno

### 2.5 Pipeline de Fusión (Módulo 2.3)

El pipeline de fusión ejecuta **4 etapas secuenciales**:

1. **TemporalAligner** — Alinea ventanas de 500ms de audio y visión. Mapea `speaker_id` ↔ `person_id` mediante co-ocurrencia temporal (mín. 30% overlap) y similitud coseno de embeddings faciales (umbral 0.6).

2. **MetricsCalculator** — Produce `StudentMetrics[]` y `GroupMetrics` a partir de las ventanas alineadas y los resultados crudos de audio.

3. **VALUERubricMapper** — Convierte las métricas en puntajes VALUE (0-20) mediante las fórmulas de la Sección 1.6-1.10.

4. **ExplanationGenerator** — Genera narrativa explicativa vía OpenAI GPT-4o-mini (temperatura 0.3, máx. 1024 tokens). Fallback determinista basado en reglas si la API no está disponible.

**Stack tecnológico completo:**

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| API | FastAPI 0.115+ |
| Async tasks | Celery 5.4+ / Redis 5+ |
| DB | PostgreSQL 16+ (asyncpg + SQLAlchemy 2.0 async) |
| File storage | MinIO S3 (compatible) |
| ML/CV | PyTorch 2.x, OpenCV 4.x, MediaPipe 0.10.x, transformers 4.x |
| Audio | librosa 0.10.x, Deepgram SDK 3.x |
| LLM | OpenAI Python SDK 1.x (gpt-4o-mini) |
| Frontend | React 18 + TypeScript 5.x, Material UI 7.x, Chart.js 4.x, Redux Toolkit |
| Infra | Docker Compose (backend + frontend + redis + postgres + minio) |

---

## OE3: Construcción del prototipo del sistema cognitivo multimodal

### 3.1 Contexto del objetivo

OE3 establece: *"Construir el prototipo del sistema cognitivo multimodal utilizando un entorno de desarrollo ágil (Scrum adaptado) basado en Python, frameworks de Deep Learning y APIs de orquestación."*

### 3.2 Estructura de salida del sistema

El pipeline de fusión produce un objeto `FusionResult` que se persiste en `AnalysisSession.result_data` (JSONB):

```json
{
  "video_path": "...",
  "aligned_features": {
    "duration_seconds": 167.0,
    "windows": [
      {
        "start": 0.0, "end": 0.5,
        "speaker_id": "speaker_0", "person_id": "person_0",
        "gaze_at_camera": true, "gaze_confidence": 0.85,
        "body_orientation": 12.5, "gesture_type": "nod",
        "emotion": "attentive", "overlap_ratio": 0.8
      }
    ],
    "speaker_to_person": { "speaker_0": "person_0" },
    "person_to_speaker": { "person_0": "speaker_0" }
  },
  "group_metrics": {
    "total_students": 4,
    "participation_cv": 0.25,
    "disruptive_interruption_rate": 0.08,
    "turn_synchronization_score": 0.72,
    "avg_gaze_contact_percentage": 68.5,
    "silence_ratio": 0.12,
    "per_student_metrics": [
      {
        "student_id": "speaker_0",
        "speaking_time_seconds": 45.2, "turn_count": 12,
        "interruption_count": 2, "interrupted_count": 5,
        "participation_ratio": 0.375,
        "gaze_contact_percentage": 72.5,
        "avg_body_orientation": 15.2,
        "dominant_emotion": "attentive",
        "initiation_count": 4,
        "back_channel_count": 3
      }
    ]
  },
  "rubric_scores": {
    "contributes_to_team_meetings": 14.2,
    "facilitates_contributions": 13.8,
    "fosters_constructive_climate": 16.5,
    "responds_to_conflict": 15.1,
    "individual_contributions_outside": 0.0,
    "per_student_scores": [
      {
        "student_id": "speaker_0",
        "contributes_to_team_meetings": 15.0,
        "facilitates_contributions": 14.2,
        "fosters_constructive_climate": 17.0,
        "responds_to_conflict": 16.5,
        "individual_contributions_outside": 0.0
      }
    ]
  },
  "explanation": {
    "narrative_text": "El grupo demostró una participación...",
    "strengths": ["Participación equilibrada (CV=0.25)..."],
    "improvements": ["Reducir interrupciones disruptivas..."],
    "recommendations": ["Establecer turno rotativo de moderador..."],
    "generated_by": "openai",
    "model_used": "gpt-4o-mini",
    "topic_description": "Los estudiantes discutieron el diseño de...",
    "intervention_summaries": {
      "speaker_0": "Lideró la discusión inicial, proponiendo..."
    }
  }
}
```

### 3.3 Módulos del frontend

| Componente | Ruta/Función | Descripción |
|---|---|---|
| **Dashboard** | `GET /teacher/dashboard` | KPIs: total grupos, sesiones, CV promedio, puntaje rúbrica promedio |
| **GroupDetail** | Página de grupo | Pestañas: Resumen, Audit, Video, Docente |
| **RubricRadarChart** | Gráfico de radar | 5 ejes VALUE (0-20) con Chart.js |
| **Tabla de rúbrica** | Pestaña Resumen | Dimensiones VALUE con puntajes numéricos y promedio |
| **Formulario docente** | Pestaña Docente | Campos editables por dimensión + nota cualitativa |
| **AuditTimeline** | Pestaña Audit | Timeline de transcripciones, interrupciones y segmentos |
| **GroupComparison** | `GET /teacher/groups/{id}/comparison` | Serie temporal de puntajes de rúbrica del grupo |

### 3.4 Base de datos: esquema relevante

**Tabla `analysis_sessions`:**

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | UUID | PK |
| `group_id` | UUID | FK → groups |
| `status` | VARCHAR | pending, processing, completed, failed |
| `video_path` | VARCHAR | Ruta en MinIO o disco |
| `duration_seconds` | FLOAT | Duración del video |
| `result_data` | JSONB | Resultados acumulados: vision, audio, key_moments, smart_vision, fusion |
| `topic_description` | TEXT | Tema inferido por LLM |
| `processed_at` | TIMESTAMP | Fecha de finalización |

**Tabla `rubric_scores`:**

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | UUID | PK |
| `session_id` | UUID | FK → analysis_sessions |
| `student_id` | UUID | FK → students |
| `evaluator_type` | VARCHAR | `system` o `teacher` |
| `collaboration_score` | FLOAT | Mapeado a: Contributes to Team Meetings |
| `communication_score` | FLOAT | Mapeado a: Facilitates Contributions |
| `responsibility_score` | FLOAT | Mapeado a: Fosters Constructive Climate |
| `leadership_score` | FLOAT | Mapeado a: Responds to Conflict |
| `technical_contribution_score` | FLOAT | Mapeado a: Individual Contributions Outside |
| `overall_score` | FLOAT | Promedio de las 5 dimensiones |
| `intervention_summary` | TEXT | Resumen por estudiante (LLM) |
| `created_at` | TIMESTAMP | Fecha de creación |

> **Nota:** Los nombres de columna en la DB mantienen la nomenclatura original del sistema. El mapeo a dimensiones VALUE se realiza en la capa API (`validate.py`). Esto permite compatibilidad hacia atrás con datos existentes.

### 3.5 Flujo de validación docente

1. El docente visualiza los puntajes automáticos (`evaluator_type = "system"`).
2. Modifica cualquier dimensión (0-20) mediante el formulario en la pestaña Docente.
3. Las correcciones se almacenan como una fila separada en `rubric_scores` con `evaluator_type = "teacher"`, **sin sobrescribir** los puntajes originales.
4. El endpoint `GET /validate/{session_id}` retorna ambos conjuntos lado a lado para comparación.

---

## OE4: Diseño experimental para evaluación estadística de precisión y concordancia

### 4.1 Contexto del objetivo

OE4 establece: *"Evaluar estadísticamente el grado de precisión técnica y concordancia inter-evaluador (ICC) del sistema frente al contraste con la rúbrica VALUE aplicada por docentes, utilizando métricas objetivas de los modelos (DER < 15%, mAP > 85%, WER documentada) y pruebas no paramétricas (Shapiro-Wilk, Wilcoxon) o paramétricas según la distribución de los datos."*

La metodología de la tesis (sección 8.2.8) especifica tres técnicas:

1. **Shapiro-Wilk** — para determinar normalidad de los puntajes.
2. **ICC bajo modelo Two-way random effects, absolute agreement** — para validar concordancia IA-docente. ICC > 0.75 confirma concordancia excelente.
3. **Wilcoxon Signed-Rank o T-Student pareada** — para verificar si la IA reduce significativamente el margen de sesgo/error intra-grupal comparado con actas históricas.

### 4.2 Diseño experimental

#### 4.2.1 Población y muestra

| Parámetro | Valor propuesto |
|---|---|
| Población | Estudiantes del último ciclo de Ingeniería de Sistemas e Inteligencia Artificial, UPAO, 2026 |
| Muestra para sesiones nuevas | $n \geq 30$ sesiones de trabajo colaborativo grabadas |
| Grupos por sesión | 3-5 estudiantes |
| Duración por sesión | 10-20 minutos |
| Tarea | Resolución de un problema de ingeniería de software en equipo |
| Docentes evaluadores | 3 docentes independientes (mínimo requerido para ICC) |
| Actas históricas | Registros de notas de trabajo en equipo de 2-3 semestres anteriores (misma carrera, cursos equivalentes) |

**Justificación del tamaño muestral:** Para un ICC two-way random effects con $k = 3$ evaluadores (2 docentes + sistema como tercer evaluador), potencia $1-\beta = 0.80$, $\alpha = 0.05$, y un ICC mínimo aceptable $\rho_1 = 0.75$ bajo $H_0: \rho = 0.50$, se requieren al menos $n = 19$ sesiones (Walter, Eliasziw & Donner, 1998). Se propone $n \geq 30$ para robustez ante posibles pérdidas y para fortalecer las pruebas de contraste de medias.

#### 4.2.2 Variables del estudio

**Variables independientes:**
- Tipo de evaluador: Sistema automático (VALUERubricMapper) vs. Docente 1 vs. Docente 2 vs. Docente 3
- Dimensión de la rúbrica: D1, D2, D4, D5 (D3 excluida por no ser evaluable en sesión grabada)
- Fuente de datos: Sesiones experimentales 2026 vs. Actas históricas (2024-2025)

**Variables dependientes:**
- Puntaje asignado en cada dimensión VALUE (escala 0-20, continua)
- Dispersión intra-grupal: desviación estándar y varianza de los puntajes por sesión

**Variables de control:**
- Misma rúbrica VALUE para todos los evaluadores
- Misma sesión de video para todos los evaluadores (dentro del grupo experimental)
- Mismo instructivo de evaluación para los docentes
- Mismo tipo de asignatura y nivel para las actas históricas

#### 4.2.3 Procedimiento

```
Fase 0: LÍNEA BASE HISTÓRICA
  ├── Recopilar actas históricas de 2-3 semestres anteriores
  │   (misma carrera, cursos con evaluación de trabajo en equipo)
  ├── Extraer puntajes de rúbrica por estudiante y sesión
  └── Calcular dispersión basal: SD y varianza intra-grupal
      → Esto establece el "margen de error humano histórico"

Fase 1: RECOLECCIÓN EXPERIMENTAL
  ├── Grabar n ≥ 30 sesiones de trabajo colaborativo (2026)
  ├── Ejecutar pipeline automático → puntajes del sistema (S_sys)
  └── Entregar los mismos videos a 3 docentes independientes
      ├── Docente A evalúa con rúbrica VALUE → S_A
      ├── Docente B evalúa con rúbrica VALUE → S_B
      └── Docente C evalúa con rúbrica VALUE → S_C

Fase 2: MÉTRICAS TÉCNICAS DE MODELOS
  ├── DER: Tasa de Error de Diarización (Deepgram Nova-2)
  ├── mAP: Precisión Media Promedio (MediaPipe Face Detection)
  └── WER: Tasa de Error de Palabras (Deepgram Nova-2)

Fase 3: ANÁLISIS ESTADÍSTICO
  ├── Shapiro-Wilk: normalidad de puntajes por dimensión y evaluador
  ├── ICC(2,1): concordancia sistema-docentes (two-way random, absolute agreement)
  ├── Contraste de sesgo/error: Wilcoxon o T-Student pareada
  │   comparando dispersión del sistema vs dispersión de actas históricas
  ├── Matriz de confusión por nivel VALUE (sistema vs docentes)
  └── Bland-Altman: sesgo sistemático por dimensión

Fase 4: INTERPRETACIÓN
  ├── ¿El sistema alcanza concordancia excelente con docentes? (ICC > 0.75)
  ├── ¿El sistema reduce el margen de sesgo/error vs actas históricas? (p < 0.05)
  └── ¿En qué dimensiones hay mayor/menor concordancia?
```

#### 4.2.4 Las actas históricas como línea base

Las actas históricas cumplen un rol fundamental en el diseño: establecen el **error basal del proceso de evaluación humana tradicional**. En los semestres anteriores, los docentes evaluaban el trabajo en equipo de forma manual (sin sistema automatizado), usando criterios similares pero sin la estandarización forzada de la rúbrica VALUE.

**Qué se extrae de las actas históricas:**

| Métrica | Cálculo | Propósito |
|---|---|---|
| $\sigma^2_{\text{hist}}$ | Varianza de los puntajes de trabajo en equipo en actas históricas | Dispersión basal del error humano |
| $\text{CV}_{\text{hist}}$ | Coeficiente de variación de los puntajes históricos | Variabilidad relativa normalizada |
| $\text{Rango}_{\text{hist}}$ | Rango intercuartílico de puntajes históricos | Amplitud típica de la discrepancia entre evaluadores humanos |

**Qué se compara:**

| Comparación | Fuente A | Fuente B | Qué responde |
|---|---|---|---|
| Concordancia | Sistema (S_sys) | Docentes 2026 (S_doc) | ¿El sistema concuerda con evaluadores humanos expertos? |
| Precisión / Reducción de error | Dispersión del sistema (SD_sys) | Dispersión de actas históricas (SD_hist) | ¿El sistema reduce el margen de error comparado con la evaluación tradicional? |

> **Nota metodológica:** No se comparan directamente los puntajes numéricos del sistema contra los puntajes de actas históricas (serían poblaciones distintas). La comparación es sobre la **dispersión/error**: ¿la variabilidad de los puntajes del sistema es significativamente menor que la variabilidad observada históricamente en la evaluación manual?

### 4.3 Métricas técnicas de los modelos individuales

Estas métricas evalúan la calidad de cada componente del pipeline, independientemente de la rúbrica:

#### 4.3.1 DER — Diarization Error Rate (Deepgram Nova-2)

**Definición:** Mide el error en la asignación de segmentos de habla a hablantes.

$$\text{DER} = \frac{\text{FA} + \text{MISS} + \text{SPK\_ERR}}{\text{TOTAL\_REF}}$$

Donde:
- FA (False Alarm): segmentos donde el sistema detecta habla pero no hay
- MISS: segmentos donde hay habla pero el sistema no detecta
- SPK_ERR: segmentos asignados al hablante incorrecto
- TOTAL_REF: duración total de habla en la referencia

**Umbral objetivo:** DER < 15%
**Método de validación:** Anotación manual de 5 sesiones (ground truth) por un anotador experto.

#### 4.3.2 mAP — Mean Average Precision (MediaPipe Face Detection)

**Definición:** Precisión de detección de rostros a diferentes umbrales de IoU.

$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i$$

**Umbral objetivo:** mAP > 85% (IoU ≥ 0.5)
**Método de validación:** Anotación manual de 100 frames aleatorios de 5 sesiones.

#### 4.3.3 WER — Word Error Rate (Deepgram Nova-2)

**Definición:** Error en la transcripción palabra por palabra.

$$\text{WER} = \frac{S + D + I}{N}$$

Donde $S$ = sustituciones, $D$ = deleciones, $I$ = inserciones, $N$ = total de palabras en referencia.

**Umbral objetivo:** WER documentada (sin umbral fijo; depende del dominio acústico)
**Método de validación:** Transcripción manual de 5 fragmentos de 2 minutos por un transcriptor experto.

### 4.4 Análisis estadístico

#### 4.4.1 Prueba de normalidad: Shapiro-Wilk

**Hipótesis:**
- $H_0$: Los puntajes siguen una distribución normal
- $H_1$: Los puntajes no siguen una distribución normal

**Procedimiento:** Aplicar Shapiro-Wilk a los puntajes de cada dimensión (D1, D2, D4, D5) para cada evaluador (Sistema, Docente A, B, C). Total: 4 dimensiones × 4 evaluadores = 16 pruebas.

Adicionalmente, aplicar Shapiro-Wilk a los puntajes extraídos de las actas históricas para verificar si la distribución basal es normal.

**Decisión:** Si $p > 0.05$ en las pruebas relevantes, se usan pruebas paramétricas. Si $p \leq 0.05$, se usan pruebas no paramétricas.

**Tabla de presentación de resultados:**

| Dimensión | Evaluador | W | p | ¿Normal? |
|---|---|---|---|---|
| D1: Contributes | Sistema | 0.96 | 0.32 | Sí |
| D1: Contributes | Docente A | 0.94 | 0.15 | Sí |
| D1: Contributes | Docente B | 0.91 | 0.08 | Sí |
| D1: Contributes | Docente C | 0.93 | 0.12 | Sí |
| D2: Facilitates | Sistema | 0.88 | 0.03 | No |
| ... | ... | ... | ... | ... |
| Actas históricas (global) | — | 0.95 | 0.21 | Sí |

#### 4.4.2 Validación de concordancia: ICC — Intraclass Correlation Coefficient

**Definición:** El ICC mide la concordancia absoluta entre múltiples evaluadores que puntúan los mismos sujetos. A diferencia de la correlación de Pearson (que mide asociación), el ICC penaliza diferencias sistemáticas en magnitud (si un evaluador consistentemente puntúa 2 puntos más alto que otro, el ICC baja; Pearson no).

**Modelo:** **ICC(2,1) — Two-way random effects, single rater, absolute agreement.**

Este modelo fue seleccionado porque:
- **Two-way random effects:** Los docentes evaluadores se consideran una muestra aleatoria de una población más amplia de posibles evaluadores. Esto permite generalizar los resultados: si el sistema concuerda con estos 3 docentes, se espera que concuerde con otros docentes de características similares.
- **Single rater (k=1):** Se reporta la confiabilidad de UN evaluador individual (no el promedio de k evaluadores). Esto es más conservador y relevante para el caso de uso real, donde típicamente un solo docente evalúa una sesión.
- **Absolute agreement:** Se exige que los puntajes sean idénticos en valor absoluto, no solo que covaríen en la misma dirección (consistency). Esto es más estricto y apropiado para una rúbrica de evaluación donde la magnitud del puntaje importa.

**Fórmula (modelo two-way random, absolute agreement, single rater):**

$$\text{ICC}(2,1) = \frac{\text{MS}_S - \text{MS}_E}{\text{MS}_S + (k-1) \cdot \text{MS}_E + \frac{k}{n}(\text{MS}_R - \text{MS}_E)}$$

Donde:
- $\text{MS}_S$: Cuadrado medio entre sujetos (sesiones)
- $\text{MS}_R$: Cuadrado medio entre evaluadores (raters)
- $\text{MS}_E$: Cuadrado medio del error (residual)
- $k$: Número de evaluadores (3: sistema + 2 docentes, o 4: sistema + 3 docentes)
- $n$: Número de sujetos (sesiones)

**Implementación:** Se utilizará la biblioteca `pingouin` de Python (`pingouin.intraclass_corr`) con los parámetros `ratings='absolute'` y `model='two-way random'`, o alternativamente `scipy.stats` con cálculo manual de los cuadrados medios.

**Comparaciones a realizar:**

1. **ICC(2,1) Sistema + Docentes (4 evaluadores):** Concordancia global entre el sistema y los 3 docentes para cada dimensión. Responde: *¿el sistema es intercambiable con un evaluador humano?*

2. **ICC(2,1) Solo docentes (3 evaluadores):** Concordancia entre los 3 docentes sin el sistema. Sirve como **techo de referencia**: representa la máxima concordancia alcanzable entre humanos. Si el ICC sistema+docentes se aproxima al ICC inter-docente, el sistema es virtualmente indistinguible de un evaluador humano adicional.

3. **ICC(2,1) Sistema vs cada docente (2 evaluadores):** Concordancia del sistema con cada docente individualmente. Responde: *¿con qué docente se parece más el sistema?*

**Interpretación (Koo & Li, 2016):**

| ICC | Interpretación |
|---|---|
| < 0.50 | Concordancia pobre |
| 0.50 – 0.75 | Concordancia moderada |
| 0.75 – 0.90 | Buena concordancia |
| > 0.90 | Concordancia excelente |

**Criterio de éxito:** ICC(2,1) ≥ 0.75 en al menos 3 de las 4 dimensiones evaluables.

**Tabla de presentación de resultados:**

| Dimensión VALUE | ICC(2,1) Sist+3Doc | IC 95% | ICC(2,1) Inter-docente | Interpretación |
|---|---|---|---|---|
| D1: Contributes to Team Meetings | 0.82 | [0.71, 0.90] | 0.85 | Buena |
| D2: Facilitates Contributions | 0.68 | [0.52, 0.80] | 0.78 | Moderada |
| D4: Fosters Constructive Climate | 0.76 | [0.63, 0.86] | 0.80 | Buena |
| D5: Responds to Conflict | 0.71 | [0.55, 0.82] | 0.77 | Moderada |

> **Lectura de la tabla:** En D1, el ICC sistema+docentes (0.82) está muy cerca del ICC inter-docente (0.85). Esto significa que agregar el sistema como cuarto evaluador no degrada significativamente la concordancia del panel — el sistema se comporta de forma similar a un docente adicional.

#### 4.4.3 Contraste de medias: precisión y reducción del margen de sesgo/error

**Objetivo:** Verificar si el sistema reduce significativamente el margen de sesgo/error intra-grupal comparado con las actas históricas, tal como lo establece la metodología de la tesis (sección 8.2.8, punto 3).

**Estrategia de comparación:**

El "margen de sesgo/error intra-grupal" se operacionaliza como la **desviación estándar (SD) de los puntajes dentro de cada grupo/sesión**. Una SD alta indica que los estudiantes del mismo grupo recibieron puntajes muy dispares (posible sesgo del evaluador). Una SD baja indica evaluación más consistente dentro del grupo.

**Procedimiento:**

1. **Calcular SD intra-grupal para el sistema:** Para cada una de las $n$ sesiones experimentales, calcular la desviación estándar de los puntajes de los estudiantes dentro de esa sesión, según el sistema. Esto produce $n$ valores de SD_sys.

2. **Calcular SD intra-grupal para actas históricas:** Para cada sesión/acta histórica disponible, calcular la desviación estándar de los puntajes de los estudiantes dentro de esa acta. Esto produce $m$ valores de SD_hist.

3. **Comparar las SD:** Aplicar prueba de contraste de medias para determinar si la SD promedio del sistema es significativamente menor que la SD promedio de las actas históricas.

**Prueba específica:**

| Condición | Prueba |
|---|---|
| Ambas muestras normales (Shapiro-Wilk $p > 0.05$) | T-Student para muestras independientes (unilateral: $\mu_{\text{sys}} < \mu_{\text{hist}}$) |
| Al menos una muestra no normal | U de Mann-Whitney (unilateral) |

> **Nota:** Aquí se usa prueba para muestras **independientes** (no pareadas), porque las actas históricas y las sesiones experimentales son poblaciones distintas. A diferencia del ICC y la matriz de confusión (que comparan sistema vs docentes sobre las MISMAS sesiones), esta prueba compara la dispersión del sistema contra una línea base histórica externa.

**Hipótesis:**
- $H_0$: $\text{SD}_{\text{sys}} \geq \text{SD}_{\text{hist}}$ — El sistema NO reduce el margen de error (la dispersión del sistema es igual o mayor que la histórica)
- $H_1$: $\text{SD}_{\text{sys}} < \text{SD}_{\text{hist}}$ — El sistema REDUCE significativamente el margen de error intra-grupal

**Tabla de presentación de resultados:**

| Dimensión | SD_sys (media ± DE) | SD_hist (media ± DE) | Prueba | Estadístico | p | ¿Reduce error? |
|---|---|---|---|---|---|---|
| D1 | 2.1 ± 0.8 | 3.5 ± 1.2 | t = −4.52 | — | <0.001 | Sí *** |
| D2 | 2.4 ± 1.0 | 3.2 ± 1.1 | U = 285 | — | 0.008 | Sí ** |
| D4 | 1.8 ± 0.7 | 2.9 ± 1.3 | t = −3.89 | — | <0.001 | Sí *** |
| D5 | 2.6 ± 1.1 | 3.1 ± 1.0 | U = 310 | — | 0.042 | Sí * |

#### 4.4.4 Contraste complementario: diferencia sistema vs docentes (mismas sesiones)

Como análisis complementario al contraste contra actas históricas, se verifica también si existen diferencias significativas entre los puntajes del sistema y los docentes **sobre las mismas sesiones experimentales**.

**Hipótesis:**
- $H_0$: $\mu_{\text{sys}} = \mu_{\text{doc}}$ — No hay diferencia entre sistema y docentes
- $H_1$: $\mu_{\text{sys}} \neq \mu_{\text{doc}}$ — Existe diferencia

**Prueba:** Depende del resultado de Shapiro-Wilk sobre las diferencias pareadas ($\Delta_i = S_{\text{sys},i} - \bar{S}_{\text{doc},i}$):

| Condición | Prueba |
|---|---|
| $\Delta_i$ normales | T-Student pareada bilateral |
| $\Delta_i$ no normales | Wilcoxon Signed-Rank bilateral |

**Corrección por comparaciones múltiples:** Bonferroni. Con 4 dimensiones evaluables (D1, D2, D4, D5): $\alpha_{\text{corregido}} = 0.05 / 4 = 0.0125$.

**Tabla de presentación de resultados:**

| Dim. | $\bar{X}_{\text{sys}}$ (SD) | $\bar{X}_{\text{doc}}$ (SD) | $\Delta$ | Estadístico | p | p corr. | ¿Signif.? |
|---|---|---|---|---|---|---|---|
| D1 | 14.2 (3.1) | 14.8 (2.9) | −0.6 | W = 145 | 0.23 | 0.92 | No |
| D2 | 13.8 (3.5) | 15.2 (3.0) | −1.4 | W = 98 | 0.04 | 0.16 | No |
| D4 | 16.5 (2.2) | 16.1 (2.5) | +0.4 | t = 0.89 | 0.38 | 1.00 | No |
| D5 | 15.1 (3.3) | 14.5 (3.6) | +0.6 | W = 180 | 0.41 | 1.00 | No |

> **Interpretación deseada:** Ninguna dimensión muestra diferencias significativas tras corrección Bonferroni, confirmando que el sistema produce puntajes estadísticamente indistinguibles de los docentes.

#### 4.4.5 Matriz de confusión por nivel VALUE

**Objetivo:** Evaluar la concordancia a nivel categórico entre el sistema y los docentes.

**Procedimiento:**
1. Convertir puntajes continuos (0-20) a niveles VALUE discretos:
   - [0, 5) → Benchmark (1)
   - [5, 10) → Milestone 2 (2)
   - [10, 15) → Milestone 3 (3)
   - [15, 20] → Capstone (4)
2. Para cada sesión y dimensión, el nivel "docente" es la moda o mediana redondeada de los 3 docentes.
3. Construir matriz de confusión: nivel sistema vs nivel docente.
4. Calcular precisión global y **Kappa de Cohen (κ)**.

**Tabla de presentación (ejemplo para D1):**

|  | Docente Benchmark | Docente Milestone 2 | Docente Milestone 3 | Docente Capstone |
|---|---|---|---|---|
| **Sist. Benchmark** | 18 | 2 | 0 | 0 |
| **Sist. Milestone 2** | 3 | 22 | 4 | 0 |
| **Sist. Milestone 3** | 0 | 3 | 19 | 3 |
| **Sist. Capstone** | 0 | 0 | 2 | 20 |

| Métrica | Valor |
|---|---|
| Precisión global | 82.3% |
| Kappa de Cohen (κ) | 0.76 |

**Interpretación de Kappa (Landis & Koch, 1977):**
- κ > 0.80: Concordancia casi perfecta
- 0.61 – 0.80: Concordancia sustancial
- 0.41 – 0.60: Concordancia moderada
- 0.21 – 0.40: Concordancia aceptable

#### 4.4.6 Análisis de sesgo: Bland-Altman

**Objetivo:** Visualizar el sesgo sistemático entre el sistema y los docentes.

**Procedimiento (por dimensión):**
1. Eje X: Promedio (sistema + media_docentes) / 2
2. Eje Y: Diferencia (sistema − media_docentes)
3. Línea de sesgo: $\bar{d}$ (media de las diferencias)
4. Límites de concordancia (95%): $\bar{d} \pm 1.96 \cdot \text{SD}(d)$

**Interpretación:**
- $\bar{d} \approx 0$ → sin sesgo sistemático
- Límites estrechos → buena concordancia
- Patrón en embudo → heterocedasticidad

### 4.5 Resumen de hipótesis del estudio (OE4)

| ID | Hipótesis | Tipo | Prueba | Criterio de éxito |
|---|---|---|---|---|
| H1 | El sistema alcanza concordancia excelente con docentes | Concordancia | ICC(2,1) two-way random, absolute agreement | ICC ≥ 0.75 en ≥ 3/4 dimensiones |
| H2 | El sistema reduce significativamente el margen de error intra-grupal vs actas históricas | Precisión / Reducción de error | T-Student independiente o U de Mann-Whitney (unilateral) | $p < 0.05$ en ≥ 3/4 dimensiones |
| H3 | No hay diferencia significativa entre puntajes del sistema y docentes | Equivalencia | Wilcoxon o T-Student pareada + Bonferroni | $p > 0.0125$ en todas las dimensiones |
| H4 | Los modelos individuales cumplen métricas técnicas mínimas | Rendimiento técnico | DER, mAP, WER | DER < 15%, mAP > 85% |
| H5 | El sistema clasifica correctamente los niveles VALUE | Concordancia categórica | Kappa de Cohen | κ ≥ 0.61 (sustancial) |
| H6 | El sesgo del sistema no es sistemático | Sesgo | Bland-Altman | $\|\bar{d}\| \leq 2$ puntos en cada dimensión |

---

## OE5: Instrumento de aceptabilidad y usabilidad (TAM)

### 5.1 Contexto del objetivo

OE5 establece: *"Determinar la aceptabilidad y usabilidad del sistema por parte del cuerpo docente evaluador mediante la aplicación del instrumento psicométrico TAM."*

### 5.2 Modelo TAM (Technology Acceptance Model)

El TAM (Davis, 1989) postula que la aceptación de una tecnología está determinada por dos constructos principales:

```
                    ┌──────────────────┐
                    │ Perceived         │
                    │ Usefulness (PU)   │──┐
                    │ Utilidad Percibida│  │
                    └──────────────────┘  │    ┌──────────────────┐    ┌──────────────┐
                                          ├───→│ Attitude Toward  │───→│ Behavioral   │
                                          │    │ Using (ATU)      │    │ Intention (BI)│
                    ┌──────────────────┐  │    │ Actitud hacia el │    │ Intención de │
                    │ Perceived Ease    │  │    │ Uso              │    │ Uso          │
                    │ of Use (PEOU)     │──┘    └──────────────────┘    └──────────────┘
                    │ Facilidad de Uso  │
                    └──────────────────┘
```

### 5.3 Instrumento propuesto

Se utilizará una adaptación del instrumento TAM validado por Davis (1989) y ampliado por Venkatesh & Davis (2000), contextualizado al sistema SMATC-UPAO. Escala Likert de 7 puntos (1 = Totalmente en desacuerdo, 7 = Totalmente de acuerdo).

**Constructo PU — Perceived Usefulness (6 ítems):**

| ID | Ítem |
|---|---|
| PU1 | Usar SMATC-UPAO me permitiría evaluar el trabajo en equipo más rápidamente |
| PU2 | Usar SMATC-UPAO mejoraría mi efectividad como evaluador de trabajo colaborativo |
| PU3 | SMATC-UPAO me ayudaría a identificar problemas de colaboración que podría pasar por alto |
| PU4 | SMATC-UPAO haría más objetiva mi evaluación del trabajo en equipo |
| PU5 | Encontraría a SMATC-UPAO útil en mi práctica docente |
| PU6 | Los puntajes de rúbrica generados por SMATC-UPAO son consistentes con mi criterio de evaluación |

**Constructo PEOU — Perceived Ease of Use (6 ítems):**

| ID | Ítem |
|---|---|
| PEOU1 | Aprender a usar SMATC-UPAO sería fácil para mí |
| PEOU2 | La interfaz de SMATC-UPAO es clara y comprensible |
| PEOU3 | Me resultaría fácil volverme hábil usando SMATC-UPAO |
| PEOU4 | La interacción con SMATC-UPAO es flexible y no requiere pasos innecesarios |
| PEOU5 | El proceso de subir un video y obtener resultados es intuitivo |
| PEOU6 | No necesitaría asistencia técnica frecuente para usar SMATC-UPAO |

**Constructo ATU — Attitude Toward Using (3 ítems):**

| ID | Ítem |
|---|---|
| ATU1 | Tengo una actitud positiva hacia el uso de SMATC-UPAO |
| ATU2 | Me gusta la idea de usar un sistema automatizado para apoyar la evaluación |
| ATU3 | Considero que SMATC-UPAO es una herramienta valiosa para la docencia universitaria |

**Constructo BI — Behavioral Intention to Use (3 ítems):**

| ID | Ítem |
|---|---|
| BI1 | Tengo la intención de usar SMATC-UPAO en mis próximos cursos |
| BI2 | Recomendaría SMATC-UPAO a otros docentes |
| BI3 | Preferiría usar SMATC-UPAO sobre la evaluación manual tradicional |

### 5.4 Procedimiento de aplicación

1. **Momento:** Inmediatamente después de que cada docente complete la evaluación de las sesiones con la rúbrica VALUE manual (Fase 1 de OE4).
2. **Participantes:** Los mismos 3 docentes que participaron como evaluadores en OE4.
3. **Modalidad:** Cuestionario digital (Google Forms o similar).
4. **Duración estimada:** 5-10 minutos.
5. **Consentimiento informado:** Incluir en el encabezado del cuestionario.

### 5.5 Análisis estadístico

#### 5.5.1 Confiabilidad del instrumento — Alpha de Cronbach

$$\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^{k} \sigma_i^2}{\sigma_t^2}\right)$$

Donde $k$ es el número de ítems, $\sigma_i^2$ es la varianza del ítem $i$, y $\sigma_t^2$ es la varianza total.

**Criterio:** $\alpha \geq 0.70$ para cada constructo (Nunnally, 1978).

**Tabla de presentación:**

| Constructo | Ítems | α de Cronbach | Interpretación |
|---|---|---|---|
| PU — Utilidad Percibida | 6 | 0.89 | Buena |
| PEOU — Facilidad de Uso | 6 | 0.85 | Buena |
| ATU — Actitud hacia el Uso | 3 | 0.78 | Aceptable |
| BI — Intención de Uso | 3 | 0.82 | Buena |

#### 5.5.2 Estadística descriptiva por constructo

| Constructo | Media | DE | Mediana | Mín | Máx |
|---|---|---|---|---|---|
| PU | 5.8 | 0.9 | 6.0 | 4.0 | 7.0 |
| PEOU | 5.5 | 1.1 | 5.5 | 3.5 | 7.0 |
| ATU | 6.0 | 0.8 | 6.0 | 4.5 | 7.0 |
| BI | 5.7 | 1.0 | 6.0 | 4.0 | 7.0 |

> **Interpretación deseada:** Medias > 5.0 (en escala 1-7) indican aceptabilidad positiva.

#### 5.5.3 Prueba de hipótesis TAM

**Hipótesis TAM clásicas:**

| Hipótesis | Relación | Prueba |
|---|---|---|
| H1_TAM | PEOU → PU (la facilidad predice utilidad) | Regresión lineal simple |
| H2_TAM | PU → ATU (la utilidad predice actitud) | Regresión lineal simple |
| H3_TAM | PEOU → ATU (la facilidad predice actitud) | Regresión lineal simple |
| H4_TAM | PU → BI (la utilidad predice intención de uso) | Regresión lineal simple |
| H5_TAM | ATU → BI (la actitud predice intención de uso) | Regresión lineal simple |

**Tabla de presentación:**

| Hipótesis | β | R² | p | ¿Soportada? |
|---|---|---|---|---|
| H1_TAM: PEOU → PU | 0.62 | 0.38 | 0.003 | Sí |
| H2_TAM: PU → ATU | 0.71 | 0.50 | <0.001 | Sí |
| H3_TAM: PEOU → ATU | 0.45 | 0.20 | 0.015 | Sí |
| H4_TAM: PU → BI | 0.58 | 0.34 | 0.008 | Sí |
| H5_TAM: ATU → BI | 0.66 | 0.44 | 0.002 | Sí |

> **Nota:** Con solo 3 docentes, la potencia estadística es limitada. Esto es inherente al diseño con pocos evaluadores expertos. Se recomienda reportar los resultados como exploratorios y complementar con análisis cualitativo (comentarios abiertos de los docentes).

---

## Limitaciones del sistema

### L1. Dimensión no evaluable (D3)
La dimensión **Individual Contributions Outside of Team Meetings** no puede evaluarse a partir de la grabación de una sesión. Requiere integración con sistemas externos (LMS, repositorios Git, plataformas de entrega).

### L2. Calidad de la detección de emociones
CLIP zero-shot no fue entrenado específicamente para expresiones faciales en entornos académicos. Iluminación, ángulos y oclusiones afectan la precisión, impactando la Dimensión 4 (*Fosters Constructive Climate*).

### L3. Ausencia de NLP semántico profundo
No se analiza el contenido semántico de las intervenciones. Esto limita la evaluación de si un estudiante "construye sobre ideas ajenas" (D2) o "hace avanzar la discusión" (D1).

### L4. Dependencia de APIs cloud
La diarización y transcripción dependen de Deepgram Nova-2, introduciendo latencia, costos y dependencia de conectividad.

### L5. Sesgo cultural y lingüístico
Los modelos (CLIP, Deepgram) fueron entrenados con datos predominantemente en inglés y poblaciones no latinoamericanas. Las normas de turn-taking y expresiones faciales varían culturalmente.

### L6. Resolución temporal
Las ventanas de 500ms pueden perder micro-interacciones relevantes para las Dimensiones 2, 4 y 5.

---

## Trabajo futuro

1. **NLP semántico:** Incorporar LLMs para analizar contenido de transcripciones: construcción sobre ideas ajenas, preguntas dirigidas, invitaciones a participar.
2. **Integración D3:** Conectar con LMS (Moodle, Canvas) y repositorios (GitHub, GitLab) para evaluar contribuciones individuales fuera de sesiones.
3. **Modelos de emoción especializados:** Fine-tuning de modelos de expresión facial en contexto académico colaborativo con datos de UPAO.
4. **Análisis de secuencias de conflicto:** Detección de resolución de conflictos, no solo inicio (interrupción disruptiva → cesión de turno, acuerdo, mediación).
5. **Calibración empírica:** Ajustar pesos $w_i$ mediante los datos del estudio OE4 (regresión contra puntajes docentes).

---

## Referencias

1. AAC&U. (2009). *VALUE Rubric for Teamwork*. American Association of Colleges and Universities.
2. Rhodes, T. L. (Ed.). (2010). *Assessing Outcomes and Improving Achievement: Tips and Tools for Using Rubrics*. AAC&U.
3. Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307-310.
4. Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, 13(3), 319-340.
5. Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155-163.
6. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.
7. Nunnally, J. C. (1978). *Psychometric Theory* (2nd ed.). McGraw-Hill.
8. Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3/4), 591-611.
9. Venkatesh, V., & Davis, F. D. (2000). A theoretical extension of the technology acceptance model. *Management Science*, 46(2), 186-204.
10. Walter, S. D., Eliasziw, M., & Donner, A. (1998). Sample size and optimal designs for reliability studies. *Statistics in Medicine*, 17(1), 101-110.
11. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
12. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172.
13. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). arXiv:2103.00020.
14. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR 2015.
15. Park, T. J., et al. (2022). A review of speaker diarization: Recent advances with deep learning. *Computer Speech & Language*, 72, 101317.

---

*Documento técnico generado el 28 de mayo de 2026. Versión 2.1. Estructurado por objetivos específicos de tesis.*
