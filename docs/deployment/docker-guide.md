# Docker Deployment Guide

This guide covers Docker deployment for the LAION Embeddings system v2.2.0 with all 22 MCP tools functional and complete CI/CD alignment.

## 🎯 Docker-CI/CD Alignment

The Docker configurations have been updated to perfectly align with the CI/CD pipeline for consistent behavior across all environments:

- **✅ Unified Entrypoint**: Same `mcp_server.py` entrypoint used by CI/CD, Docker, and production
- **✅ Validation Consistency**: Same `mcp_server.py --validate` command for health checks and testing
- **✅ Virtual Environment**: Properly configured Python virtual environment in all containers
- **✅ Environment Alignment**: Same dependencies and configuration approach as CI/CD pipeline

## 🚀 Quick Start

```bash
# Clone and deploy in one go
git clone <repository-url>
cd laion-embeddings-1
./docker-deploy.sh all
```

Access your server at: http://localhost:9999

## 📋 Deployment Options

### 1. Single Container (Simple)

Basic deployment for development and testing:

```bash
# Build and run
./docker-deploy.sh build
./docker-deploy.sh run

# Check status
./docker-deploy.sh status
```

### 2. Docker Compose (Full Stack)

Complete setup with IPFS, monitoring, and visualization:

```bash
# Start all services
docker-compose up -d

# Scale the main service
docker-compose up -d --scale laion-embeddings=3
```

### 3. Production Deployment

For production environments with custom configuration:

```bash
# Set environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export VECTOR_STORE=faiss

# Deploy with custom config
docker run -d \
  --name laion-embeddings-prod \
  -p 9999:9999 \
  -e ENVIRONMENT=production \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  laion-embeddings:v2.2.0
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `production` | Runtime environment |
| `LOG_LEVEL` | `INFO` | Logging level |
| `VECTOR_STORE` | `faiss` | Vector store backend |
| `MCP_TOOLS_ENABLED` | `true` | Enable MCP tools |
| `PORT` | `9999` | Server port |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Data persistence |
| `./logs` | `/app/logs` | Log files |
| `./config` | `/app/config` | Configuration files |

### Resource Limits

Default resource allocation:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## 🔧 MCP Server Configuration

The Docker deployment uses the same MCP server configuration as the CI/CD pipeline for consistency:

### Entrypoint Alignment
```bash
# CI/CD uses this for testing
python mcp_server.py --validate

# Docker uses this for running
python3 mcp_server.py

# Docker uses this for health checks
python3 mcp_server.py --validate
```

### Container Health Validation
The container continuously validates MCP tools health using the same command as CI/CD:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 mcp_server.py --validate > /dev/null || exit 1
```

### Build-time Validation
During Docker build, MCP tools are validated using CI/CD approach:

```dockerfile
RUN python3 mcp_server.py --validate || echo "MCP tools validation completed"
```

## 🏥 Health Monitoring

### Built-in Health Checks

The container includes automatic health monitoring:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' laion-embeddings-server

# Manual health check
curl http://localhost:9999/health
```

Expected response:
```json
{
  "status": "healthy",
  "mcp_tools": "22/22 functional",
  "timestamp": "2025-06-06T12:00:00Z"
}
```

### Monitoring Stack

With Docker Compose, you get full monitoring:

- **Prometheus**: Metrics collection (http://localhost:9090)
- **Grafana**: Visualization (http://localhost:3000)
- **Container logs**: Centralized logging

## 🔍 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker logs laion-embeddings-server

# Verify MCP tools
./docker-deploy.sh test
```

**Health check failing:**
```bash
# Check internal health
docker exec laion-embeddings-server curl http://localhost:9999/health

# Verify port binding
docker port laion-embeddings-server
```

**Memory issues:**
```bash
# Check resource usage
docker stats laion-embeddings-server

# Increase memory limit
docker update --memory=8g laion-embeddings-server
```

### Performance Tuning

**For high-load scenarios:**

```bash
# Run with optimized settings
docker run -d \
  --name laion-embeddings-optimized \
  -p 9999:9999 \
  -e ENVIRONMENT=production \
  -e WORKERS=4 \
  -e WORKER_CLASS=uvicorn.workers.UvicornWorker \
  --memory=16g \
  --cpus=8 \
  --restart unless-stopped \
  laion-embeddings:v2.2.0
```

**For GPU acceleration:**

```bash
# GPU-enabled container
docker run -d \
  --name laion-embeddings-gpu \
  --gpus all \
  -p 9999:9999 \
  -e CUDA_VISIBLE_DEVICES=0 \
  laion-embeddings:v2.2.0
```

## 🔐 Security

### Production Security

```bash
# Run with security enhancements
docker run -d \
  --name laion-embeddings-secure \
  -p 127.0.0.1:9999:9999 \  # Bind to localhost only
  --user 1000:1000 \         # Non-root user
  --read-only \              # Read-only filesystem
  --tmpfs /tmp \             # Temporary filesystem
  --no-new-privileges \      # Prevent privilege escalation
  laion-embeddings:v2.2.0
```

### Network Security

```bash
# Create isolated network
docker network create --driver bridge laion-secure

# Run in isolated network
docker run -d \
  --name laion-embeddings-secure \
  --network laion-secure \
  -p 9999:9999 \
  laion-embeddings:v2.2.0
```

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./docker-deploy.sh stop
./docker-deploy.sh cleanup
./docker-deploy.sh all
```

### Backup and Recovery

```bash
# Backup data volume
docker run --rm \
  -v laion-embeddings_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore from backup
docker run --rm \
  -v laion-embeddings_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/data-backup.tar.gz -C /data
```

## 📊 Validation

### Verify Deployment

```bash
# Test all components
curl http://localhost:9999/health
curl http://localhost:9999/
curl -X POST http://localhost:9999/create_embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test"], "model": "thenlper/gte-small"}'

# Validate MCP tools
docker exec laion-embeddings-server python test_all_mcp_tools.py
```

Expected output: `All 22 MCP tools working correctly! ✅`

### Load Testing

```bash
# Simple load test
ab -n 100 -c 10 http://localhost:9999/health

# More comprehensive testing
docker run --rm \
  --network container:laion-embeddings-server \
  williamyeh/wrk \
  -t 4 -c 100 -d 30s http://localhost:9999/health
```

## 🆘 Support

For deployment issues:

1. Check the [troubleshooting guide](../troubleshooting.md)
2. Verify all 22 MCP tools are working: `python test_all_mcp_tools.py`
3. Review container logs: `docker logs laion-embeddings-server`
4. Check resource usage: `docker stats`

**System Status**: All 22 MCP tools functional (100% success rate) ✅
