# 🐳 Docker & Rancher Setup Guide

**Production-ready containerization with Rancher orchestration.**

---

## Part 1: Docker Installation

### Windows

1. **Download Docker Desktop:** https://www.docker.com/products/docker-desktop
2. **Install & Enable WSL 2**
3. **Verify:**
   ```powershell
   docker --version
   docker compose --version
   ```

### macOS

```bash
brew install docker --cask
# Or download from: https://www.docker.com/products/docker-desktop
```

### Linux (Ubuntu)

```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
```

---

## Part 2: Build & Run with Docker Compose

### Basic Commands

```bash
# Build all services
docker compose build

# Start services (background)
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f frontend

# Stop services
docker compose down

# Remove volumes (clean data)
docker compose down -v
```

### Service Structure

```yaml
services:
  api:           # FastAPI backend (port 8000)
  frontend:      # Streamlit UI (port 8501)
  train-stage1:  # GPU training (Stage 1) - run with profile
  train-stage2:  # GPU training (Stage 2) - run with profile
```

### Start Training Inside Docker (GPU)

Requires NVIDIA Container Runtime: https://github.com/NVIDIA/nvidia-docker

```bash
# Build with GPU support
docker compose build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Run training
docker compose --profile training up train-stage1
docker compose --profile training up train-stage2

# Check GPU usage inside container
docker exec <container_id> nvidia-smi
```

---

## Part 3: Rancher (Container Orchestration & UI)

**Rancher** = UI for managing Docker containers, monitoring, and deployments.

### Install Rancher (Docker)

```bash
# Latest Rancher on port 8080
docker run -d --restart=unless-stopped \
  -p 8080:80 -p 8443:443 \
  --name rancher \
  rancher/rancher:latest
```

**Access:** https://localhost:8080

**First login:** 
- Username: `admin`
- Password: Check logs with `docker logs rancher` for auto-generated password

### Add Local Cluster to Rancher

1. **Rancher UI** → Clusters → Create
2. Select **Local** cluster (your machine)
3. Follow instructions to register
4. Now you can manage all containers from Rancher UI

### Deploy RareSight via Rancher

#### Option A: Using Docker Compose (Simple)

1. **Rancher UI** → Workloads → Compose
2. Paste `docker-compose.yml` content
3. Click Deploy
4. Monitor from Rancher dashboard

#### Option B: Using Kubernetes (Advanced)

1. Install K3s (lightweight Kubernetes):
   ```bash
   curl -sfL https://get.k3s.io | sh -
   ```

2. Create Kubernetes manifests (`k8s/deployment.yaml`):
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: raresight-api
   spec:
     selector:
       app: raresight
     ports:
       - port: 8000
         targetPort: 8000
     type: LoadBalancer
   ---
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: raresight-api
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: raresight
     template:
       metadata:
         labels:
           app: raresight
       spec:
         containers:
           - name: api
             image: raresight:latest
             ports:
               - containerPort: 8000
             volumeMounts:
               - name: checkpoints
                 mountPath: /app/checkpoints
         volumes:
           - name: checkpoints
             hostPath:
               path: /path/to/checkpoints
   ```

3. Deploy:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

4. **Rancher UI** → Import cluster → Add to Rancher

---

## Part 4: Monitoring & Performance

### Using Rancher Dashboard

- **Workloads:** View running containers
- **Nodes:** Monitor CPU, memory usage
- **Logs:** Real-time container logs
- **Exec:** Shell into container

### Using Portainer (Lightweight Alternative)

```bash
docker run -d -p 8000:8000 -p 9000:9000 \
  --name portainer \
  --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  portainer/portainer-ce:latest
```

**Access:** http://localhost:9000

---

## Part 5: Production Checklist

- [ ] Use `raresight:gpu` image for GPU training
- [ ] Enable health checks in docker-compose
- [ ] Set resource limits (memory, CPU)
- [ ] Use persistent volumes for data/checkpoints
- [ ] Setup logging (ELK stack or Prometheus)
- [ ] Use environment file (`.env`) for secrets
- [ ] Setup CI/CD (GitHub Actions → Docker Hub)

### Example Health Check

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Resource Limits

```yaml
services:
  api:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

---

## Part 6: GPU Training in Docker

### Prerequisites

- NVIDIA GPU
- NVIDIA drivers installed
- NVIDIA Container Runtime

### Build & Run

```bash
# Build with GPU support
docker compose build --build-arg CUDA_VERSION=12.1 api

# Run training with GPU
docker compose run --rm \
  --gpus all \
  train-stage1

# Verify GPU inside container
docker run --gpus all nvidia/cuda:12.1.1-base nvidia-smi
```

---

## Quick Command Reference

```bash
# Build
docker compose build

# Start (foreground)
docker compose up

# Start (background)
docker compose up -d

# View logs
docker compose logs -f

# Shell into service
docker exec -it <service_name> bash

# Stop & remove
docker compose down

# Clean everything
docker compose down -v --remove-orphans
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change ports in docker-compose.yml |
| Container won't start | Check logs: `docker compose logs <service>` |
| GPU not found | Install NVIDIA Container Runtime |
| Out of disk | Run `docker system prune` |
| Permission denied | Use `sudo` or add user to docker group |

---

## Next Steps

1. ✅ Install Docker Desktop
2. ✅ Run `docker compose up`
3. ✅ Access API: http://localhost:8000
4. ✅ Access Frontend: http://localhost:8501
5. ✅ Install Rancher for UI management
6. ✅ Deploy via Rancher dashboard

**Questions?** Check Docker docs: https://docs.docker.com
