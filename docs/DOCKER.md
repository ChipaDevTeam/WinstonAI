# Docker Guide for WinstonAI

This guide explains how to run WinstonAI using Docker containers with GPU support.

## Prerequisites

- Docker 20.10 or higher
- Docker Compose 1.29 or higher
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support

## Installing NVIDIA Docker

### Ubuntu/Debian

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

### Verify Installation

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Build the Image

```bash
docker build -t winston-ai:latest .
```

### Run Training

```bash
docker run --gpus all -v $(pwd)/models:/app/models winston-ai:latest
```

### Run Live Trading

```bash
docker run --gpus all \
  -e POCKETOPTION_EMAIL=your_email@example.com \
  -e POCKETOPTION_PASSWORD=your_password \
  -v $(pwd)/models:/app/models \
  winston-ai:latest python src/ultra_live_trading_bot.py
```

## Using Docker Compose

### Configuration

Create a `.env` file:

```bash
POCKETOPTION_EMAIL=your_email@example.com
POCKETOPTION_PASSWORD=your_password
```

### Start Services

```bash
# Start training
docker-compose up winston-train

# Start trading bot
docker-compose up winston-trade

# Start all services
docker-compose up -d
```

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f winston-trade
```

## Volume Mounts

The Docker setup mounts the following directories:

- `./src` → `/app/src` - Source code
- `./models` → `/app/models` - Trained models
- `./logs` → `/app/logs` - Log files

Changes to source code are reflected immediately in the container.

## GPU Configuration

### Use Specific GPU

```bash
docker run --gpus '"device=0"' winston-ai:latest
```

### Use Multiple GPUs

```bash
docker run --gpus '"device=0,1"' winston-ai:latest
```

### Limit GPU Memory

```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --memory=16g \
  winston-ai:latest
```

## Development with Docker

### Interactive Shell

```bash
docker run --gpus all -it --rm -v $(pwd):/app winston-ai:latest bash
```

### Run Tests

```bash
docker run --gpus all -v $(pwd):/app winston-ai:latest pytest tests/
```

### Install Additional Packages

```bash
# Enter container
docker-compose exec winston-train bash

# Install package
pip install package-name
```

## Production Deployment

### Build Production Image

```bash
docker build -t winston-ai:production -f Dockerfile .
```

### Push to Registry

```bash
# Tag image
docker tag winston-ai:production registry.example.com/winston-ai:latest

# Push
docker push registry.example.com/winston-ai:latest
```

### Deploy

```bash
# Pull and run
docker pull registry.example.com/winston-ai:latest
docker run --gpus all -d \
  --name winston-production \
  --restart always \
  -v /path/to/models:/app/models \
  -v /path/to/logs:/app/logs \
  registry.example.com/winston-ai:latest
```

## Monitoring

### Container Stats

```bash
docker stats winston-ai-trading
```

### GPU Usage

```bash
docker exec winston-ai-training nvidia-smi
```

### View Logs

```bash
# Real-time logs
docker logs -f winston-ai-trading

# Last 100 lines
docker logs --tail 100 winston-ai-training
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
# Should contain:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }
```

### Out of Memory

Reduce batch size in configuration or increase Docker memory limit:

```bash
docker run --gpus all --memory=16g winston-ai:latest
```

### Permission Issues

```bash
# Run with current user
docker run --gpus all --user $(id -u):$(id -g) -v $(pwd):/app winston-ai:latest
```

### Slow Build

Use multi-stage build and Docker buildkit:

```bash
DOCKER_BUILDKIT=1 docker build -t winston-ai:latest .
```

## Best Practices

1. **Use volumes** for models and logs to persist data
2. **Set resource limits** to prevent container from using all resources
3. **Use environment variables** for configuration
4. **Enable auto-restart** for production containers
5. **Monitor logs** regularly
6. **Update base images** regularly for security patches
7. **Use specific tags** instead of `latest` in production

## Multi-Stage Build (Advanced)

For smaller production images:

```dockerfile
# Build stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder
# ... install dependencies ...

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /app /app
# ... minimal runtime setup ...
```

## Docker Hub

Pre-built images may be available:

```bash
docker pull chipadevteam/winston-ai:latest
docker run --gpus all chipadevteam/winston-ai:latest
```

## See Also

- [Installation Guide](INSTALLATION.md)
- [Configuration Guide](CONFIGURATION.md)
- [Main README](../README.md)

---

For more help, see the [GitHub Issues](https://github.com/ChipaDevTeam/WinstonAI/issues).
