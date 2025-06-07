# Troubleshooting

This guide covers common issues you might encounter when using the LAION Embeddings system, particularly with the IPFS and DuckDB vector stores.

## 🎉 Recent Updates (v2.2.0)

**Good News!** As of v2.2.0, all major MCP tool issues have been resolved:
- ✅ All 22 MCP tools are now fully functional (100% success rate)
- ✅ Import errors have been systematically fixed
- ✅ Method signature inconsistencies resolved
- ✅ Type errors and runtime problems eliminated
- ✅ Comprehensive test validation completed

**If you're upgrading from an earlier version**, most tool-related issues listed below have been automatically resolved. Run the validation script to confirm:
```bash
python test_all_mcp_tools.py
```

## Installation Issues

### Missing Dependencies

**Issue**: Installation fails due to missing dependencies.

**Solution**:
1. Make sure you've run the dependency installer:
   ```bash
   ./install_depends.sh
   ```

2. For IPFS-specific dependencies:
   ```bash
   pip install ipfs-kit-py>=0.9.0
   ```

3. For DuckDB-specific dependencies:
   ```bash
   pip install duckdb>=0.9.0 pyarrow>=14.0.0
   ```

### IPFS Node Connection Failure

**Issue**: Cannot connect to IPFS node.

**Solution**:
1. Verify the IPFS node is running:
   ```bash
   ipfs id
   ```

2. Check API accessibility:
   ```bash
   curl http://localhost:5001/api/v0/version
   ```

3. Configure CORS settings:
   ```bash
   ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
   ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "POST", "GET"]'
   ```

4. Restart the IPFS daemon:
   ```bash
   ipfs daemon
   ```

## Vector Store Issues

### Wrong Vector Dimensions

**Issue**: Error about incompatible vector dimensions.

**Solution**:
1. Check the dimension of your vectors:
   ```python
   print(f"Vector dimension: {len(vector)}")
   ```

2. Create index with matching dimensions:
   ```python
   await store.create_index(dimension=384)  # Match your model's output dimension
   ```

3. Verify model output dimensions:
   ```python
   from services.embedding import get_embedding_model
   
   model = get_embedding_model("thenlper/gte-small")
   embedding = await model.embed("Test text")
   print(f"Model output dimension: {len(embedding)}")
   ```

### Slow Search Performance

**Issue**: Vector searches are too slow.

**Solution**:

#### For IPFS Vector Store:

1. Enable caching:
   ```yaml
   vector_databases:
     ipfs:
       enable_cache: true
       cache_size_mb: 1024
   ```

2. Optimize block size:
   ```yaml
   vector_databases:
     ipfs:
       block_size: 262144  # 256KB
   ```

3. Use vector quantization:
   ```yaml
   vector_databases:
     ipfs:
       quantization:
         method: "pq"
         subquantizers: 8
   ```

#### For DuckDB Vector Store:

1. Increase memory allocation:
   ```yaml
   vector_databases:
     duckdb:
       memory_limit: "8GB"
   ```

2. Optimize Parquet configuration:
   ```yaml
   vector_databases:
     duckdb:
       row_group_size: 10000
       compression: "zstd"
   ```

3. Enable prepared statements:
   ```yaml
   vector_databases:
     duckdb:
       use_prepared_statements: true
   ```

### Out of Memory Errors

**Issue**: System crashes with out-of-memory errors.

**Solution**:
1. Enable sharding to distribute memory usage:
   ```yaml
   vector_databases:
     ipfs:  # or duckdb
       sharding:
         enabled: true
         shard_count: 8
   ```

2. Use vector quantization to reduce memory footprint:
   ```yaml
   vector_databases:
     ipfs:  # or duckdb
       quantization:
         method: "sq"
         bits: 8
   ```

3. For DuckDB, adjust memory limits:
   ```yaml
   vector_databases:
     duckdb:
       memory_limit: "4GB"
   ```

4. Process data in smaller batches:
   ```python
   batch_size = 1000  # Reduce if still having memory issues
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       await store.add_documents(batch)
   ```

## IPFS-Specific Issues

### IPFS Content Not Found

**Issue**: IPFS returns "content not found" errors.

**Solution**:
1. Check if content is properly pinned:
   ```bash
   ipfs pin ls | grep <CID>
   ```

2. If not pinned, pin the content:
   ```bash
   ipfs pin add <CID>
   ```

3. Make sure the IPFS node has enough storage:
   ```bash
   ipfs repo stat
   ```

4. Configure the store to automatically pin content:
   ```python
   from services.providers.ipfs_store import IPFSVectorStore
   
   store = IPFSVectorStore(
       # Other parameters...
       pin=True,
       pin_recursive=True
   )
   ```

### Slow IPFS Operations

**Issue**: IPFS operations are very slow.

**Solution**:
1. Check IPFS daemon health:
   ```bash
   ipfs diag sys
   ```

2. Run garbage collection on IPFS repo:
   ```bash
   ipfs repo gc
   ```

3. Connect to faster/closer IPFS nodes:
   ```python
   from services.providers.ipfs_store import IPFSVectorStore
   
   store = IPFSVectorStore(
       # Other parameters...
       ipfs_nodes=[
           "http://faster-node:5001",
           "http://closer-node:5001"
       ]
   )
   ```

4. Enable aggressive caching:
   ```python
   from services.providers.ipfs_store import IPFSVectorStore
   from ipfs_kit_py.arcache import ARCache
   
   cache = ARCache(max_size_bytes=2_000_000_000)  # 2GB cache
   
   store = IPFSVectorStore(
       # Other parameters...
       cache=cache
   )
   ```

## DuckDB-Specific Issues

### DuckDB Database Locked

**Issue**: DuckDB returns "database is locked" errors.

**Solution**:
1. Ensure only one process is accessing the database:
   ```python
   from services.providers.duckdb_store import DuckDBVectorStore
   
   store = DuckDBVectorStore(
       # Other parameters...
       read_only=True  # Use when multiple processes need read access
   )
   ```

2. Close connections properly:
   ```python
   await store.disconnect()  # Always disconnect when done
   ```

3. Use a different database path for each process:
   ```python
   from services.providers.duckdb_store import DuckDBVectorStore
   
   store = DuckDBVectorStore(
       # Other parameters...
       database_path=f"vectors_process_{os.getpid()}.duckdb"
   )
   ```

### DuckDB Performance Degradation

**Issue**: DuckDB performance decreases over time.

**Solution**:
1. Vacuum the database periodically:
   ```python
   from services.providers.duckdb_store import DuckDBVectorStore
   
   store = DuckDBVectorStore()
   await store.connect()
   
   # Execute vacuum
   await store.execute_sql("VACUUM")
   ```

2. Use appropriate indexes:
   ```python
   from services.providers.duckdb_store import DuckDBVectorStore
   
   store = DuckDBVectorStore()
   await store.connect()
   
   # Create indexes for frequently filtered columns
   await store.execute_sql("CREATE INDEX idx_metadata_field ON vectors(metadata_field)")
   ```

3. Optimize Parquet files:
   ```python
   from services.providers.duckdb_store import DuckDBVectorStore
   
   store = DuckDBVectorStore(
       # Other parameters...
       optimize_parquet=True
   )
   ```

## API and Configuration Issues

### API Endpoints Return Errors

**Issue**: API endpoints return unexpected errors.

**Solution**:
1. Check the server logs:
   ```bash
   tail -f laion_embeddings.log
   ```

2. Enable debug mode:
   ```bash
   export DEBUG=1
   ./run.sh
   ```

3. Verify configuration file syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/vector_databases.yaml'))"
   ```

### Configuration Not Applied

**Issue**: Changes to configuration files not taking effect.

**Solution**:
1. Restart the server to apply configuration changes:
   ```bash
   ./run.sh
   ```

2. Verify configuration file path:
   ```python
   from services.vector_config import load_config
   
   config = load_config("config/vector_databases.yaml")
   print(config)
   ```

3. Check for environment variables overriding configuration:
   ```bash
   env | grep VECTOR
   ```

## Testing and Validation

### Tests Failing

**Issue**: Vector store tests failing.

**Solution**:
1. Run tests with more verbose output:
   ```bash
   python -m pytest test_vector_stores.py -v
   ```

2. Check for missing dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Run tests with environment variables for specific backends:
   ```bash
   VECTOR_STORE=ipfs python -m pytest test_vector_advanced.py
   ```

### Missing Vector Provider

**Issue**: Vector store provider not available.

**Solution**:
1. Verify the provider is registered:
   ```python
   from services.vector_store_factory import get_available_providers
   
   providers = get_available_providers()
   print(providers)
   ```

2. Check for import errors:
   ```python
   try:
       from services.providers.ipfs_store import IPFSVectorStore
       print("IPFS provider available")
   except ImportError as e:
       print(f"IPFS provider not available: {e}")
   ```

3. Install missing dependencies:
   ```bash
   pip install ipfs-kit-py>=0.9.0  # For IPFS
   pip install duckdb>=0.9.0 pyarrow>=14.0.0  # For DuckDB
   ```

## Architectural Issues

### Integration with Other Systems

**Issue**: Difficulty integrating with other systems.

**Solution**:
1. Use the vector store base interface for consistent API:
   ```python
   from services.vector_store_base import VectorStoreBase
   from services.vector_store_factory import create_vector_store
   
   store: VectorStoreBase = await create_vector_store()
   ```

2. Export data for external systems:
   ```python
   # For DuckDB
   await duckdb_store.execute_sql("COPY (SELECT * FROM vectors) TO 'export.csv' (FORMAT CSV, HEADER)")
   
   # For IPFS
   cid = await ipfs_store.export_to_ipfs()
   print(f"Data exported to IPFS: {cid}")
   ```

### Data Migration

**Issue**: Need to migrate data between vector stores.

**Solution**:
1. Use the migration utility:
   ```python
   from services.vector_migration import migrate_vector_store
   
   source_store = await create_vector_store(db_type=VectorDBType.FAISS)
   target_store = await create_vector_store(db_type=VectorDBType.IPFS)
   
   # Migrate data
   await migrate_vector_store(source_store, target_store)
   ```

2. Manual export and import:
   ```python
   # Export from source
   documents = await source_store.get_all_documents()
   
   # Import to target
   await target_store.add_documents(documents)
   ```

## Advanced Troubleshooting

### Diagnostic Tool

For more comprehensive diagnostics, use the built-in diagnostic tool:

```bash
python -m services.diagnostics --vector-store=ipfs
```

This will:
1. Check all dependencies
2. Verify configuration
3. Test basic operations
4. Measure performance
5. Generate a diagnostic report

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=1
export VECTOR_STORE_LOG_LEVEL=DEBUG
./run.sh
```

### Performance Profiling

Profile performance issues:

```python
import cProfile
import pstats

# Profile a function
def profile_search():
    cProfile.runctx('store.search(query)', globals(), locals(), 'search_stats')
    
    # Print stats
    p = pstats.Stats('search_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats(20)
```

## Getting Help

If you continue to experience issues:

1. Check the [FAQ](faq.md) for common questions and answers
2. Look for similar issues in the project repository
3. Run the diagnostic tool and share the output
4. Provide reproducible test case when asking for help
