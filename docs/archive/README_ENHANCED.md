# LAION Embeddings API - Enhanced Version

A powerful, production-ready FastAPI-based service for creating, managing, and searching embeddings with distributed storage capabilities.

## 🚀 New Features & Improvements

### ✅ Recently Implemented (Phase 1)

- **🔒 JWT Authentication & Authorization** - Role-based access control with admin/user permissions
- **📊 Comprehensive Monitoring** - Prometheus metrics, health checks, and system monitoring
- **⚡ Intelligent Caching** - In-memory cache with TTL for improved performance
- **🛡️ Rate Limiting** - IP-based rate limiting to prevent abuse
- **📝 Structured Logging** - JSON-formatted logging with request tracking
- **✅ Input Validation** - Robust input sanitization and validation
- **🔧 Enhanced Error Handling** - Proper HTTP exceptions with detailed error messages
- **🧪 Comprehensive Testing** - Unit tests, integration tests, and performance benchmarks
- **⚙️ CI/CD Pipeline** - GitHub Actions with automated testing and security scanning
- **📚 API Documentation** - Enhanced OpenAPI docs with authentication support

### 🏗️ Architecture Improvements

- **Type Safety**: Full Pydantic model validation with proper type annotations
- **Async/Await**: Proper async implementation throughout the codebase
- **Middleware Stack**: Rate limiting, metrics collection, and request tracking
- **Background Tasks**: Enhanced background task management with error handling
- **Security**: JWT-based authentication with role-based permissions

## 🔐 Authentication

The API now supports JWT-based authentication:

### Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user",
    "password": "user123"
  }'
```

### Using JWT Token
```bash
# Use the token in subsequent requests
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "machine learning",
    "collection": "my-collection",
    "n": 10
  }'
```

### Default Users
- **Admin**: username: `admin`, password: `admin123` (full access)
- **User**: username: `user`, password: `user123` (read access)

## 📊 Monitoring & Observability

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/health/detailed
```

### Metrics
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# JSON metrics
curl http://localhost:8000/metrics/json
```

### Cache Management
```bash
# Cache statistics
curl http://localhost:8000/cache/stats

# Clear expired cache entries
curl -X POST http://localhost:8000/cache/clear
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd laion-embeddings-1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables** (optional)
```bash
export JWT_SECRET_KEY="your-secret-key-for-production"
```

4. **Run the application**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

```bash
# Build the image
docker build -t laion-embeddings .

# Run the container
docker run -p 8000:8000 laion-embeddings
```

## 📚 API Endpoints

### Core Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/` | GET | No | API information |
| `/health` | GET | No | Basic health check |
| `/health/detailed` | GET | No | Detailed health with metrics |

### Authentication

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/auth/login` | POST | No | Login and get JWT token |
| `/auth/me` | GET | Yes | Get current user info |

### Embeddings Operations

| Endpoint | Method | Auth Required | Permission | Description |
|----------|--------|---------------|------------|-------------|
| `/search` | POST | Yes | Read | Search embeddings |
| `/create_embeddings` | POST | Yes | Write | Create new embeddings |
| `/load` | POST | Yes | Write | Load embeddings |
| `/add_endpoint` | POST | Yes | Admin | Add model endpoints |

### Background Tasks

| Endpoint | Method | Auth Required | Permission | Description |
|----------|--------|---------------|------------|-------------|
| `/shard_embeddings` | POST | Yes | Write | Shard embeddings |
| `/index_sparse_embeddings` | POST | Yes | Write | Index sparse embeddings |
| `/index_cluster` | POST | Yes | Write | Index clusters |
| `/storacha_clusters` | POST | Yes | Write | Storacha operations |

### Monitoring

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/metrics` | GET | No | Prometheus metrics |
| `/metrics/json` | GET | No | JSON metrics |
| `/cache/stats` | GET | No | Cache statistics |
| `/cache/clear` | POST | No | Clear expired cache |

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest test/unit/ -v

# Integration tests  
pytest test/integration/ -v

# Performance tests
pytest test/performance/ -v --benchmark-only

# All tests with coverage
pytest test/ -v --cov=. --cov-report=html
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🔧 Development

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting
- **bandit**: Security scanning

### Run Development Tools
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy . --ignore-missing-imports

# Linting
flake8 .

# Security scan
bandit -r .
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: IP-based request limiting (100 requests/minute)
- **Input Validation**: Comprehensive input sanitization
- **Security Headers**: Proper CORS and security headers
- **Dependency Scanning**: Automated vulnerability scanning

## 📈 Performance Features

- **Intelligent Caching**: TTL-based caching for search results
- **Background Processing**: Non-blocking background tasks
- **Connection Pooling**: Efficient database connections
- **Metrics Collection**: Real-time performance monitoring
- **Memory Management**: Adaptive batching and memory monitoring

## 🏗️ Project Structure

```
laion-embeddings-1/
├── main.py                 # Main FastAPI application
├── auth.py                # Authentication and authorization
├── monitoring.py          # Metrics and monitoring
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── test/                 # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── performance/     # Performance tests
│   └── fixtures/        # Test fixtures
├── .github/workflows/   # CI/CD pipeline
└── docs/               # Documentation
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure you're using the correct username/password
   - Check that JWT token is included in Authorization header

2. **Rate Limiting**
   - Default limit is 100 requests/minute per IP
   - Use authentication to get higher limits

3. **Memory Issues**
   - Monitor `/metrics/json` for memory usage
   - Clear cache regularly with `/cache/clear`

### Getting Help

1. Check the detailed health endpoint: `/health/detailed`
2. Review logs for error messages
3. Check metrics for performance issues
4. Verify authentication tokens are valid

## 🛣️ Roadmap

### Phase 2 (Next 2 weeks)
- [ ] Redis-based distributed caching
- [ ] PostgreSQL metadata storage
- [ ] Advanced monitoring dashboard
- [ ] Auto-scaling capabilities

### Phase 3 (Next month)
- [ ] Multi-model ensemble support
- [ ] Advanced search features
- [ ] Data pipeline optimization
- [ ] Extended API documentation

## 🌐 IPFS Integration - Now Fully Tested and Reliable

The IPFS integration has been significantly improved with the following key enhancements:

### 📊 Distributed Vector Storage

Our system now reliably stores and retrieves vector embeddings through IPFS for truly decentralized search:

- **Sharded Architecture**: Automatically partitions large vector collections into optimally-sized shards
- **Manifest Management**: Tracks vector distribution across the network with consistent manifests
- **Fault Tolerance**: Continues functioning despite node failures or network issues
- **Metadata Association**: Preserves rich metadata alongside vector embeddings

### 🔧 Fixed Issues

Recent maintenance has resolved several critical issues:

- **✅ Type Handling**: Improved numpy array conversions for reliable vector storage and retrieval
- **✅ Parameter Management**: Fixed parameter ordering in core storage methods
- **✅ Metadata Preservation**: Ensured metadata consistency through storage operations
- **✅ Error Propagation**: Better error handling and reporting for IPFS operations
- **✅ Testing**: Comprehensive test suite with 100% pass rate

### 📝 Documentation

Detailed documentation for the IPFS integration is available at:

- [IPFS Vector Service Documentation](docs/ipfs-vector-service.md) - Complete guide to the IPFS integration
- [IPFS Integration Examples](docs/examples/ipfs-examples.md) - Working examples for common use cases

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and ensure code quality
4. Submit a pull request

Please ensure all tests pass and code follows the project standards before submitting.
