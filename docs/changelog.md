# Changelog

All notable changes to the LAION Embeddings project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- IPFS vector store integration with full distributed capabilities
- DuckDB vector store with Parquet storage support
- Vector quantization support for all vector stores
- Advanced sharding capabilities across all providers
- Comprehensive documentation for all vector stores
- Integration examples for IPFS and DuckDB

## [2.1.0] - 2025-06-01

### Added
- IPFS vector store integration
  - Content-addressed vector storage
  - Distributed search capabilities
  - IPFS-specific sharding implementation
  - ARCache integration for high-performance caching
- DuckDB vector store implementation
  - Parquet file format support
  - SQL-based filtering and advanced queries
  - Analytical capabilities integration
- Vector quantization framework
  - Scalar quantization (SQ)
  - Product quantization (PQ)
  - Optimized product quantization (OPQ)
- Enhanced sharding capabilities
  - Hash-based sharding
  - Range-based sharding
  - Consistent hashing
  - Directory-based sharding (IPFS)
- New documentation
  - Vector stores overview
  - IPFS vector service guide
  - DuckDB vector service guide
  - Vector quantization guide
  - Sharding guide
  - Performance optimization guide
  - Basic usage examples
  - IPFS examples
  - DuckDB examples

### Changed
- Refactored vector store base interface for better extensibility
- Improved vector store factory with dependency checks
- Enhanced configuration system with environment variable override support
- Updated all documentation to reflect new features

### Fixed
- Vector normalization handling across all providers
- Memory management in large vector collections
- Thread safety in concurrent vector operations
- Error handling and reporting in distributed operations

## [2.0.0] - 2025-01-15

### Added
- Unified vector store interface
- FAISS and HNSW vector store implementations
- Vector store factory pattern
- Basic sharding support
- Initial documentation structure

### Changed
- Complete architecture redesign with provider pattern
- Improved error handling and reporting
- Enhanced configuration system
- Updated API endpoints for vector operations

## [1.2.0] - 2024-11-05

### Added
- Support for additional embedding models
- Basic vector filtering capabilities
- Improved logging system
- Initial IPFS integration planning

### Fixed
- Memory leaks in vector search operations
- Concurrent operation thread safety
- API endpoint error handling

## [1.1.0] - 2024-09-20

### Added
- Multi-model embedding support
- FastAPI endpoint improvements
- Initial vector store abstraction
- Basic documentation

### Changed
- Performance improvements for vector search
- Enhanced API response format
- Better error handling

## [1.0.0] - 2024-07-01

### Added
- Initial release of LAION Embeddings
- Basic embedding generation and search
- FAISS vector store integration
- FastAPI endpoints for basic operations
- Simple documentation
