#!/bin/bash
# COGNO EC2 Setup Script
# Uso: Ejecutar dentro de la instancia EC2 (Ubuntu 22.04)

set -e

echo "=========================================="
echo "  COGNO - Setup Script para EC2"
echo "=========================================="

# Variables
PROJECT_DIR="/home/ubuntu/cogno"
REPO_URL="https://github.com/TU_USUARIO/cogno.git"  # Cambiar

# 1. Actualizar sistema
echo "[1/8] Actualizando sistema..."
apt update && apt upgrade -y

# 2. Instalar Docker
echo "[2/8] Instalando Docker..."
apt install -y ca-certificates curl gnupg lsb-release
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 3. Agregar usuario ubuntu al grupo docker
echo "[3/8] Configurando permisos Docker..."
usermod -aG docker ubuntu

# 4. Instalar Docker Compose standalone
echo "[4/8] Instalando Docker Compose..."
curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 5. Clonar repositorio
echo "[5/8] Clonando repositorio..."
cd /home/ubuntu
if [ -d "$PROJECT_DIR" ]; then
    echo "El directorio ya existe, haciendo pull..."
    cd $PROJECT_DIR && git pull origin main
else
    git clone $REPO_URL $PROJECT_DIR
fi

# 6. Crear archivo .env
echo "[6/8] Creando archivo .env..."
cat > $PROJECT_DIR/infrastructure/.env << 'EOF'
# Database
DATABASE_URL=postgresql://smatc:smatc123@postgres:5432/smatc

# Redis
REDIS_URL=redis://redis:6379/0

# S3/MinIO
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin123
S3_BUCKET=videos

# Paths
UPLOAD_TMP_DIR=/shared/uploads
FRAMES_DIR=/shared/uploads/frames

# AI - OpenAI (obtener de https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-tu-key-aqui

# AI - Deepgram (obtener de https://console.deepgram.com/signup)
DEEPGRAM_API_KEY=dg_tu_key
DEEPGRAM_MODEL=nova-2

# Ollama
OLLAMA_BASE_URL=https://api.ollama.com
OLLAMA_API_KEY=
OLLAMA_VISION_MODEL=qwen3-vl:8b

# Whisper
WHISPER_MODEL_SIZE=medium
WHISPER_LANGUAGE=es
EOF

# 7. Crear directorio compartido
echo "[7/8] Creando directorios..."
mkdir -p /shared/uploads/frames
chmod -R 777 /shared

# 8. Cambiar permisos
echo "[8/8] Configurando permisos..."
chown -R ubuntu:ubuntu $PROJECT_DIR

echo ""
echo "=========================================="
echo "  Setup completo!"
echo "=========================================="
echo ""
echo "Próximos pasos:"
echo "1. Editar $PROJECT_DIR/infrastructure/.env con tus API keys"
echo "2. cd $PROJECT_DIR/infrastructure"
echo "3. docker compose up -d --build"
echo "4. docker compose logs -f  (ver logs)"
echo ""
echo "La API estará disponible en: http://<IP-EC2>:8000"
echo "=========================================="