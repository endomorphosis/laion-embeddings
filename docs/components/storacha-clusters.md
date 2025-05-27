# Storacha Clusters Component

The Storacha Clusters component provides integration with Storacha (formerly Web3.Storage) for decentralized storage of embeddings and datasets. It enables seamless backup, synchronization, and distribution of embedding data across the decentralized web.

## Overview

The `storacha_clusters` class provides:
- **Decentralized Storage**: Store embeddings on the distributed web via Storacha
- **Data Synchronization**: Sync content between IPFS clusters and Storacha
- **Backup Management**: Automated backup strategies for critical data
- **Content Distribution**: Distribute embeddings globally via Content Delivery Networks

## Key Features

### Web3 Storage Integration
- **Storacha API**: Direct integration with Storacha storage services
- **IPFS Compatibility**: Seamless integration with existing IPFS workflows
- **CAR File Support**: Efficient Content Addressable aRchive handling
- **Metadata Preservation**: Maintain data provenance and metadata

### Distributed Backup
- **Automatic Backup**: Schedule regular backups to Storacha
- **Incremental Sync**: Only upload changed or new content
- **Redundancy**: Multiple storage providers for high availability
- **Verification**: Verify backup integrity and accessibility

### Performance Optimization
- **Batch Operations**: Efficient handling of large datasets
- **Compression**: Automatic compression for reduced storage costs
- **Parallel Uploads**: Concurrent uploads for faster synchronization
- **CDN Distribution**: Global content distribution via CDN

## Usage

### Basic Storacha Integration

```python
from storacha_clusters import storacha_clusters

# Configuration
metadata = {
    "sync_config": {
        "interval": 3600,  # 1 hour sync interval
        "batch_size": 100,
        "compression": True
    },
    "backup_strategy": "incremental",
    "retention_policy": {
        "keep_versions": 5,
        "max_age_days": 365
    }
}

resources = {
    "storacha_token": "your-storacha-token",
    "ipfs_gateway": "http://localhost:8080",
    "cluster_api": "http://localhost:9094"
}

# Initialize component
storacha = storacha_clusters(resources, metadata)

# Run component tests
test_results = storacha.test()
print(f"Component status: {test_results}")
```

### Content Synchronization

```python
# Sync IPFS cluster content to Storacha
async def sync_to_storacha():
    """Synchronize cluster content to Storacha storage"""
    
    # Get all pinned content from IPFS cluster
    pinset = storacha.ipfs_kit_py.ipfs_get_pinset()
    
    sync_report = {
        "total_items": len(pinset),
        "uploaded": 0,
        "skipped": 0,
        "failed": 0,
        "errors": []
    }
    
    for cid in pinset:
        try:
            # Check if already uploaded
            if await storacha.content_exists_in_storacha(cid):
                sync_report["skipped"] += 1
                continue
            
            # Fetch content from IPFS
            content = storacha.ipfs_kit_py.ipfs_get(cid)
            
            # Upload to Storacha
            upload_result = await storacha.upload_to_storacha(cid, content)
            
            if upload_result["success"]:
                sync_report["uploaded"] += 1
            else:
                sync_report["failed"] += 1
                sync_report["errors"].append({
                    "cid": cid,
                    "error": upload_result["error"]
                })
                
        except Exception as e:
            sync_report["failed"] += 1
            sync_report["errors"].append({
                "cid": cid,
                "error": str(e)
            })
    
    return sync_report

# Execute synchronization
sync_result = await sync_to_storacha()
print(f"Sync complete: {sync_result}")
```

### Advanced Backup Strategies

```python
# Implement tiered backup strategy
backup_tiers = {
    "hot": {
        "storage": "ipfs_cluster",
        "retention": "30_days",
        "access_pattern": "frequent"
    },
    "warm": {
        "storage": "storacha",
        "retention": "1_year", 
        "access_pattern": "occasional"
    },
    "cold": {
        "storage": "storacha_archive",
        "retention": "indefinite",
        "access_pattern": "rare"
    }
}

async def tiered_backup(content_cid, content_type="embedding"):
    """Implement tiered backup based on content type and age"""
    
    # Determine backup tier based on content metadata
    content_age = await storacha.get_content_age(content_cid)
    access_frequency = await storacha.get_access_frequency(content_cid)
    
    if content_age < 30 and access_frequency > 10:
        tier = "hot"
    elif content_age < 365:
        tier = "warm"  
    else:
        tier = "cold"
    
    # Execute backup to appropriate tier
    backup_config = backup_tiers[tier]
    result = await storacha.backup_to_tier(content_cid, backup_config)
    
    return {"tier": tier, "result": result}
```

## Configuration

### Metadata Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `sync_config` | dict | Synchronization settings | No |
| `backup_strategy` | str | Backup approach (full/incremental) | No |
| `retention_policy` | dict | Data retention configuration | No |
| `compression` | bool | Enable content compression | No |

### Sync Configuration

```python
sync_config = {
    "interval": 3600,           # Sync interval in seconds
    "batch_size": 100,          # Items per batch
    "compression": True,        # Enable compression
    "verify_uploads": True,     # Verify uploaded content
    "retry_failed": True,       # Retry failed uploads
    "max_retries": 3           # Maximum retry attempts
}
```

### Retention Policy

```python
retention_policy = {
    "keep_versions": 5,         # Number of versions to keep
    "max_age_days": 365,       # Maximum age in days
    "cleanup_interval": 7,      # Cleanup check interval (days)
    "archive_old": True,       # Archive instead of delete
    "compression_threshold": 30 # Compress files older than 30 days
}
```

## API Reference

### Class: `storacha_clusters`

#### `__init__(resources, metadata)`
Initialize the Storacha clusters component.

**Parameters:**
- `resources` (dict): Storacha API credentials and endpoints
- `metadata` (dict): Configuration for sync and backup operations

#### `test()`
Run comprehensive tests on all component functionality.

**Returns:**
- Dictionary with test results for each subsystem:
  - `test_ipfs_kit_init`: IPFS kit initialization status
  - `test_ipfs_kit`: IPFS functionality test
  - `test_ipfs_parquet_to_car`: CAR file conversion test
  - `test_storacha_clusters`: Storacha integration test

## Implementation Examples

### Automated Backup Pipeline

```python
# Create automated backup pipeline
class AutomatedBackupPipeline:
    def __init__(self, storacha_component):
        self.storacha = storacha_component
        self.backup_schedule = {
            "embeddings": {"interval": 3600, "priority": "high"},
            "datasets": {"interval": 7200, "priority": "medium"},
            "models": {"interval": 14400, "priority": "low"}
        }
    
    async def run_scheduled_backups(self):
        """Execute scheduled backups based on content type"""
        
        for content_type, schedule in self.backup_schedule.items():
            try:
                # Get content of specific type
                content_list = await self.get_content_by_type(content_type)
                
                # Backup each item
                for content_cid in content_list:
                    backup_result = await self.backup_content(
                        content_cid, 
                        priority=schedule["priority"]
                    )
                    
                    if not backup_result["success"]:
                        print(f"Backup failed for {content_cid}: {backup_result['error']}")
                
            except Exception as e:
                print(f"Error backing up {content_type}: {e}")
    
    async def backup_content(self, cid, priority="medium"):
        """Backup individual content item"""
        
        try:
            # Check if backup needed
            if await self.storacha.backup_up_to_date(cid):
                return {"success": True, "action": "skipped", "reason": "up_to_date"}
            
            # Fetch content
            content = self.storacha.ipfs_kit_py.ipfs_get(cid)
            
            # Upload to Storacha with priority
            upload_config = {
                "priority": priority,
                "compression": True,
                "verify": True
            }
            
            result = await self.storacha.upload_with_config(cid, content, upload_config)
            return {"success": True, "action": "uploaded", "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Usage
pipeline = AutomatedBackupPipeline(storacha)
await pipeline.run_scheduled_backups()
```

### Content Distribution Network

```python
# Set up global content distribution
class ContentDistributionManager:
    def __init__(self, storacha_component):
        self.storacha = storacha_component
        self.cdn_regions = [
            "us-east-1", "us-west-1", "eu-west-1", 
            "asia-southeast-1", "australia-southeast-1"
        ]
    
    async def distribute_globally(self, content_cid):
        """Distribute content to multiple global regions"""
        
        distribution_report = {
            "regions": {},
            "total_regions": len(self.cdn_regions),
            "successful": 0,
            "failed": 0
        }
        
        for region in self.cdn_regions:
            try:
                # Upload to region-specific endpoint
                result = await self.storacha.upload_to_region(
                    content_cid, 
                    region
                )
                
                distribution_report["regions"][region] = {
                    "status": "success",
                    "url": result["cdn_url"],
                    "upload_time": result["upload_time"]
                }
                distribution_report["successful"] += 1
                
            except Exception as e:
                distribution_report["regions"][region] = {
                    "status": "failed",
                    "error": str(e)
                }
                distribution_report["failed"] += 1
        
        return distribution_report
    
    async def get_optimal_endpoint(self, user_location):
        """Get optimal CDN endpoint for user location"""
        
        # Simple geographic routing
        region_mapping = {
            "north_america": ["us-east-1", "us-west-1"],
            "europe": ["eu-west-1"],
            "asia": ["asia-southeast-1"],
            "oceania": ["australia-southeast-1"]
        }
        
        preferred_regions = region_mapping.get(user_location, ["us-east-1"])
        
        # Check availability and select best endpoint
        for region in preferred_regions:
            if await self.storacha.region_available(region):
                return await self.storacha.get_region_endpoint(region)
        
        # Fallback to any available region
        return await self.storacha.get_fallback_endpoint()
```

### Data Migration and Recovery

```python
# Implement data migration and disaster recovery
class DataRecoveryManager:
    def __init__(self, storacha_component):
        self.storacha = storacha_component
        
    async def full_disaster_recovery(self, recovery_target):
        """Perform full disaster recovery from Storacha backups"""
        
        recovery_report = {
            "started_at": datetime.utcnow(),
            "total_items": 0,
            "recovered": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Get all backed up content from Storacha
            backup_inventory = await self.storacha.get_backup_inventory()
            recovery_report["total_items"] = len(backup_inventory)
            
            for backup_item in backup_inventory:
                try:
                    # Download from Storacha
                    content = await self.storacha.download_backup(
                        backup_item["cid"]
                    )
                    
                    # Restore to recovery target
                    restore_result = await self.restore_content(
                        backup_item["cid"],
                        content,
                        recovery_target
                    )
                    
                    if restore_result["success"]:
                        recovery_report["recovered"] += 1
                    else:
                        recovery_report["failed"] += 1
                        recovery_report["errors"].append({
                            "cid": backup_item["cid"],
                            "error": restore_result["error"]
                        })
                        
                except Exception as e:
                    recovery_report["failed"] += 1
                    recovery_report["errors"].append({
                        "cid": backup_item["cid"],
                        "error": str(e)
                    })
            
            recovery_report["completed_at"] = datetime.utcnow()
            recovery_report["duration"] = (
                recovery_report["completed_at"] - recovery_report["started_at"]
            ).total_seconds()
            
        except Exception as e:
            recovery_report["fatal_error"] = str(e)
        
        return recovery_report
    
    async def selective_recovery(self, cid_list, recovery_target):
        """Recover specific content items"""
        
        recovery_results = {}
        
        for cid in cid_list:
            try:
                # Check if backup exists
                if not await self.storacha.backup_exists(cid):
                    recovery_results[cid] = {
                        "status": "failed",
                        "error": "No backup found"
                    }
                    continue
                
                # Download and restore
                content = await self.storacha.download_backup(cid)
                restore_result = await self.restore_content(
                    cid, content, recovery_target
                )
                
                recovery_results[cid] = restore_result
                
            except Exception as e:
                recovery_results[cid] = {
                    "status": "failed", 
                    "error": str(e)
                }
        
        return recovery_results
```

## Performance Optimization

### Batch Upload Optimization

```python
# Optimize for large-scale uploads
class BatchUploadOptimizer:
    def __init__(self, storacha_component):
        self.storacha = storacha_component
        self.batch_config = {
            "max_batch_size": 100,
            "max_concurrent": 5,
            "compression_threshold": 1024*1024,  # 1MB
            "retry_attempts": 3
        }
    
    async def optimized_batch_upload(self, content_list):
        """Upload content in optimized batches"""
        
        # Sort by size for better batching
        sorted_content = sorted(
            content_list,
            key=lambda x: x.get("size", 0)
        )
        
        # Create optimized batches
        batches = self.create_optimal_batches(sorted_content)
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(self.batch_config["max_concurrent"])
        
        async def process_batch(batch):
            async with semaphore:
                return await self.upload_batch(batch)
        
        # Execute all batches
        batch_results = await asyncio.gather(*[
            process_batch(batch) for batch in batches
        ])
        
        return self.aggregate_results(batch_results)
    
    def create_optimal_batches(self, content_list):
        """Create optimally sized batches"""
        
        batches = []
        current_batch = []
        current_size = 0
        max_batch_size = self.batch_config["max_batch_size"]
        
        for content in content_list:
            content_size = content.get("size", 0)
            
            # Start new batch if current would be too large
            if (len(current_batch) >= max_batch_size or 
                current_size + content_size > 100*1024*1024):  # 100MB limit
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
            
            current_batch.append(content)
            current_size += content_size
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
```

### Compression Strategies

```python
# Implement intelligent compression
class CompressionManager:
    def __init__(self):
        self.compression_algorithms = {
            "gzip": {"ratio": 0.7, "speed": "fast"},
            "bzip2": {"ratio": 0.6, "speed": "slow"}, 
            "lzma": {"ratio": 0.5, "speed": "very_slow"}
        }
    
    def select_compression(self, content_size, content_type):
        """Select optimal compression based on content"""
        
        # Text content compresses well
        if content_type in ["text", "json", "csv"]:
            if content_size > 10*1024*1024:  # > 10MB
                return "lzma"  # Best ratio for large text
            else:
                return "gzip"  # Good balance
        
        # Binary content (embeddings) 
        elif content_type in ["embeddings", "vectors"]:
            return "gzip"  # Fast with decent ratio
        
        # Already compressed content
        elif content_type in ["images", "videos", "archives"]:
            return None  # Skip compression
        
        return "gzip"  # Default
    
    async def compress_content(self, content, algorithm):
        """Compress content using specified algorithm"""
        
        if algorithm == "gzip":
            import gzip
            return gzip.compress(content)
        elif algorithm == "bzip2":
            import bz2
            return bz2.compress(content)
        elif algorithm == "lzma":
            import lzma
            return lzma.compress(content)
        else:
            return content  # No compression
```

## Error Handling and Monitoring

### Comprehensive Error Handling

```python
class RobustStorachaManager:
    def __init__(self, storacha_component):
        self.storacha = storacha_component
        self.error_handlers = {
            "network": self.handle_network_error,
            "auth": self.handle_auth_error,
            "quota": self.handle_quota_error,
            "timeout": self.handle_timeout_error
        }
    
    async def robust_upload(self, cid, content):
        """Upload with comprehensive error handling"""
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                result = await self.storacha.upload_to_storacha(cid, content)
                return {"success": True, "result": result}
                
            except Exception as e:
                error_type = self.classify_error(e)
                
                # Handle specific error types
                if error_type in self.error_handlers:
                    handled = await self.error_handlers[error_type](e, attempt)
                    if handled["retry"]:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return {"success": False, "error": handled["message"]}
                
                # Unknown error - retry with backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    return {"success": False, "error": str(e)}
    
    def classify_error(self, error):
        """Classify error type for appropriate handling"""
        
        error_str = str(error).lower()
        
        if "network" in error_str or "connection" in error_str:
            return "network"
        elif "auth" in error_str or "unauthorized" in error_str:
            return "auth"
        elif "quota" in error_str or "limit" in error_str:
            return "quota"
        elif "timeout" in error_str:
            return "timeout"
        else:
            return "unknown"
```

## Dependencies

- `ipfs_kit_py`: IPFS integration and operations
- `ipfs_embeddings_py`: Core embedding functionality
- `asyncio`: Asynchronous operations
- `aiohttp`: HTTP client for Storacha API
- `datetime`: Timestamp management

## Related Components

- [IPFS Cluster Index](ipfs-cluster-index.md): IPFS cluster management
- [Create Embeddings](create-embeddings.md): Content generation
- [Search Embeddings](search-embeddings.md): Content discovery
- [IPFS Integration](../ipfs/README.md): IPFS documentation
