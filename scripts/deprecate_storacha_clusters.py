#!/usr/bin/env python3
"""
Script to deprecate storacha_clusters module in favor of ipfs_kit_py.

This script will:
1. Replace storacha_clusters.py with a deprecation notice
2. Update all imports to use ipfs_kit_py instead
3. Create a migration guide for users
"""

import os
import sys
import re
import shutil
from pathlib import Path

def create_deprecation_notice():
    """Create a deprecation notice for storacha_clusters module."""
    deprecation_content = '''"""
DEPRECATED: storacha_clusters module

This module is deprecated and has been replaced by ipfs_kit_py.

Migration Guide:
---------------

OLD (deprecated):
    from storacha_clusters import storacha_clusters
    storacha = storacha_clusters(resources, metadata)
    
NEW (recommended):
    from ipfs_kit_py.storacha_kit import storacha_kit
    from ipfs_kit_py.ipfs_kit import ipfs_kit
    
    # For Storacha functionality:
    storacha = storacha_kit(api_key="your_api_key")
    
    # For general IPFS functionality:
    ipfs = ipfs_kit(resources, metadata)

Features mapping:
-----------------
- storacha_clusters.test() -> storacha_kit.validate_connection()
- storacha_clusters.kmeans_cluster_split -> ipfs_kit.faiss_kit.kmeans_cluster_split_dataset
- All IPFS operations -> ipfs_kit methods
- All S3 operations -> ipfs_kit.s3_kit methods
- All embedding operations -> ipfs_kit.ipfs_embeddings methods

For more information, see: docs/IPFS_KIT_INTEGRATION_GUIDE.md
"""

import warnings
import logging

logger = logging.getLogger(__name__)

class storacha_clusters:
    """DEPRECATED: Use ipfs_kit_py.storacha_kit instead."""
    
    def __init__(self, resources=None, metadata=None):
        warnings.warn(
            "storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead. "
            "See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "storacha_clusters is deprecated. Use ipfs_kit_py.storacha_kit instead."
        )
        
        # Initialize with None to prevent usage
        self.resources = resources
        self.metadata = metadata
        self.ipfs_kit_py = None
        self.s3_kit_py = None
        self.ipfs_embeddings_py = None
        self.ipfs_parquet_to_car = None
        self.kmeans_cluster_split = None
        
    def test(self):
        """DEPRECATED: Use storacha_kit.validate_connection() instead."""
        raise DeprecationWarning(
            "storacha_clusters.test() is deprecated. "
            "Use ipfs_kit_py.storacha_kit.validate_connection() instead."
        )
        
    def __getattr__(self, name):
        """Catch all attribute access and show deprecation warning."""
        raise AttributeError(
            f"storacha_clusters.{name} is deprecated. "
            f"Use ipfs_kit_py instead. See docs/IPFS_KIT_INTEGRATION_GUIDE.md"
        )

if __name__ == "__main__":
    print("ERROR: storacha_clusters is deprecated.")
    print("Use ipfs_kit_py instead.")
    print("See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.")
    sys.exit(1)
'''
    return deprecation_content

def update_init_py():
    """Update __init__.py with deprecation notice."""
    init_content = '''"""DEPRECATED: Storacha Clusters Module

This module is deprecated and has been replaced by ipfs_kit_py.

Use ipfs_kit_py.storacha_kit instead.
See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.
"""

import warnings

warnings.warn(
    "storacha_clusters module is deprecated. Use ipfs_kit_py.storacha_kit instead. "
    "See docs/IPFS_KIT_INTEGRATION_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

from .storacha_clusters import storacha_clusters

__all__ = ['storacha_clusters']
'''
    return init_content

def update_imports_in_file(file_path):
    """Update imports in a single file."""
    if not os.path.exists(file_path):
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    
    # Replace storacha_clusters imports
    patterns = [
        (r'from storacha_clusters import storacha_clusters', 
         'from ipfs_kit_py.storacha_kit import storacha_kit\nfrom ipfs_kit_py.ipfs_kit import ipfs_kit'),
        (r'import storacha_clusters', 
         'import ipfs_kit_py.storacha_kit as storacha_kit\nimport ipfs_kit_py.ipfs_kit as ipfs_kit'),
        (r'storacha_clusters\(([^)]*)\)', 
         r'ipfs_kit(\1)  # TODO: Review - use storacha_kit for Storacha-specific functionality'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        try:
            # Create backup
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in: {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return False

def main():
    """Main deprecation process."""
    print("Starting storacha_clusters deprecation process...")
    
    base_dir = Path(__file__).parent.parent
    storacha_dir = base_dir / "storacha_clusters"
    
    if not storacha_dir.exists():
        print(f"storacha_clusters directory not found: {storacha_dir}")
        return False
    
    # 1. Backup the original files
    backup_dir = base_dir / "storacha_clusters_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(storacha_dir, backup_dir)
    print(f"Created backup at: {backup_dir}")
    
    # 2. Replace storacha_clusters.py with deprecation notice
    storacha_py = storacha_dir / "storacha_clusters.py"
    deprecation_content = create_deprecation_notice()
    
    try:
        with open(storacha_py, 'w', encoding='utf-8') as f:
            f.write(deprecation_content)
        print(f"Replaced {storacha_py} with deprecation notice")
    except Exception as e:
        print(f"Error writing deprecation notice: {e}")
        return False
    
    # 3. Update __init__.py with deprecation warning
    init_py = storacha_dir / "__init__.py"
    init_content = update_init_py()
    
    try:
        with open(init_py, 'w', encoding='utf-8') as f:
            f.write(init_content)
        print(f"Updated {init_py} with deprecation warning")
    except Exception as e:
        print(f"Error updating __init__.py: {e}")
        return False
    
    # 4. Update imports in key files
    files_to_update = [
        base_dir / "__init__.py",
        base_dir / "main.py",
    ]
    
    updated_files = []
    for file_path in files_to_update:
        if update_imports_in_file(file_path):
            updated_files.append(file_path)
    
    print(f"\\nDeprecation process completed!")
    print(f"- Created backup: {backup_dir}")
    print(f"- Updated storacha_clusters module with deprecation notices")
    print(f"- Updated imports in {len(updated_files)} files: {updated_files}")
    print(f"\\nNext steps:")
    print(f"1. Review the updated files and test the application")
    print(f"2. Update any remaining manual references to storacha_clusters")
    print(f"3. See docs/IPFS_KIT_INTEGRATION_GUIDE.md for full migration guide")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
