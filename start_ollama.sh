#!/bin/bash
set -e

# Crear carpeta persistente para Docker
mkdir -p /home/ec2-user/SageMaker/docker-data

# Configurar Docker para usar almacenamiento persistente
cat > /etc/docker/daemon.json <<'EOF'
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "data-root": "/home/ec2-user/SageMaker/docker-data"
}
EOF

# Reiniciar Docker para aplicar la configuración
systemctl restart docker

# Esperar a que Docker esté listo
sleep 5

# Si el contenedor ollama ya existe, iniciarlo
if docker ps -a --format '{{.Names}}' | grep -q '^ollama$'; then
    docker start ollama
else
    # Si no existe, crearlo
    docker run -d \
      -v ollama:/root/.ollama \
      -p 11434:11434 \
      --name ollama \
      ollama/ollama
fi

# Esperar a que Ollama esté listo
sleep 5

