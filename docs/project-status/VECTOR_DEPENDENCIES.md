# Vector Store Provider Dependencies

To fully utilize the new vector store providers, the following dependencies need to be installed:

## IPFS/IPLD Vector Store Dependencies

The IPFS vector store provider requires either the `ipfs_kit_py` library or `ipfshttpclient`:

```bash
# Using ipfs_kit_py (preferred)
pip install ipfs_kit_py

# Alternative: using ipfshttpclient
pip install ipfshttpclient
```

Additionally, an IPFS daemon should be running and accessible. You can install and run IPFS by following instructions at https://docs.ipfs.tech/install/

## DuckDB/Parquet Vector Store Dependencies

The DuckDB vector store provider requires both DuckDB and PyArrow/Parquet:

```bash
pip install duckdb pyarrow
```

## Dependency Status in Tests

The integration tests correctly identified that the required dependencies are not installed in the current environment. This is expected behavior as the provider architecture is designed to gracefully handle missing dependencies by skipping registration of unsupported providers.

## Next Steps for Testing

To properly test these providers, you should:

1. Install the required dependencies
2. Ensure that the IPFS daemon is running (for IPFS provider)
3. Run the test scripts again

```bash
# After installing dependencies:
python test_vector_stores.py --store ipfs
python test_vector_stores.py --store duckdb
```

## Graceful Degradation

Even without these dependencies, the unified vector store architecture continues to function with the available providers. This ensures that applications using the architecture can operate in environments with varying dependencies without errors.

## Production Deployment

In a production environment, you should:

1. Include all required dependencies in your requirements.txt or setup.py
2. Add proper dependency checks in your application startup
3. Provide configuration options to select available providers
