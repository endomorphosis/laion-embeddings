#!/usr/bin/env python3
"""
Mock IPFS client for testing
"""

import json
import os
import sys
from typing import Dict, Any, List, Set, Optional
from unittest.mock import MagicMock, Mock

# Create a mock for the ipfshttpclient module
class MockPinAPI:
    """Mock for the pin API in IPFS client"""
    
    def __init__(self, client):
        self.client = client
    
    def add(self, cid, *args, **kwargs):
        """Add a pin"""
        self.client._pinned.add(cid)
        return {"Pins": [cid]}
    
    def rm(self, cid, *args, **kwargs):
        """Remove a pin"""
        if cid in self.client._pinned:
            self.client._pinned.remove(cid)
        return {"Pins": [cid]}

class MockIPFSClient:
    """Mock IPFS client implementation"""
    
    def __init__(self):
        self._storage = {}
        self._counter = 0
        self._pinned = set()
        self.pin = MockPinAPI(self)
        
        # Pre-populate with test manifests
        self._storage["QmManifest123"] = json.dumps({
            'total_vectors': 4,
            'total_shards': 2,
            'shard_size': 2,
            'dimension': 2,
            'shards': {
                'shard_0': {'ipfs_hash': 'QmShard123', 'vector_count': 2}
            }
        })
        self._storage["QmShard123"] = json.dumps({
            'vectors': [[0.1, 0.2], [0.3, 0.4]],
            'metadata': {
                'texts': ['text1', 'text2'],
                'metadata': [{}, {}]
            }
        })
    
    def add_json(self, data):
        """Add JSON data to IPFS"""
        cid = f"Qm{self._counter}"
        self._storage[cid] = json.dumps(data)
        self._counter += 1
        return cid
    
    def get_json(self, cid):
        """Get JSON data from IPFS"""
        if cid not in self._storage:
            if cid == "QmManifest123":  # Predefined test CID
                return {
                    'total_vectors': 4,
                    'total_shards': 2,
                    'shard_size': 2,
                    'dimension': 2,
                    'shards': {
                        'shard_0': {'ipfs_hash': 'QmShard123', 'vector_count': 2}
                    }
                }
            elif cid == "QmShard123":  # Predefined test CID
                return {
                    'vectors': [[0.1, 0.2], [0.3, 0.4]],
                    'metadata': {
                        'texts': ['text1', 'text2'],
                        'metadata': [{}, {}]
                    }
                }
            raise ValueError(f"CID {cid} not found")
        return json.loads(self._storage[cid])
            
    def version(self):
        """Get IPFS version"""
        return {'Version': '0.14.0'}

# Create mock module
mock_module = MagicMock()
mock_module.connect = Mock(return_value=MockIPFSClient())

# Detect if we're running tests
if os.environ.get('TESTING', '').lower() == 'true':
    # Add the mock module to sys.modules to make imports work
    sys.modules['ipfshttpclient'] = mock_module
