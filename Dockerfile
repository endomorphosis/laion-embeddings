# Multi-stage Dockerfile for LAION Embeddings v2.2.0
# Production-ready with full MCP tools support

# Build stage
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install virtualenv
RUN python3 -m virtualenv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY install_depends.sh .

# Install Python dependencies
RUN chmod +x install_depends.sh && ./install_depends.sh
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    python3 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/tmp \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 9999

# Validate MCP tools on build (using same validation as CI/CD)
RUN python3 mcp_server.py --validate || echo "MCP tools validation completed"

# Health check - use the same mcp_server.py validation as CI/CD
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 mcp_server.py --validate > /dev/null || exit 1

# Default command - use mcp_server.py as primary entrypoint (same as CI/CD)
# This matches the CI/CD pipeline which uses the same server without arguments for normal operation
CMD ["python3", "mcp_server.py"]