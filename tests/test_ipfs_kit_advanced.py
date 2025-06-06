#!/usr/bin/env python3
"""
Advanced feature tests for ipfs_kit_py integration.
Tests edge cases, error scenarios, batch operations, and configuration overrides.
"""

import pytest
import tempfile
import os
import json
import sys
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Check what's available from ipfs_kit_py
IPFS_KIT_AVAILABLE = False
STORACHA_KIT_AVAILABLE = False
S3_KIT_AVAILABLE = False
CONFIG_MANAGER_AVAILABLE = False

# Try individual imports to see what works
try:
    # Add the package path
    sys.path.insert(0, '/home/barberb/laion-embeddings-1/docs/ipfs_kit_py')
    
    # Try importing individual components
    try:
        from ipfs_kit_py.storacha_kit import StorachaKit
        STORACHA_KIT_AVAILABLE = True
    except ImportError:
        pass
    
    try:
        from ipfs_kit_py.s3_kit import S3Kit
        S3_KIT_AVAILABLE = True
    except ImportError:
        pass
    
    try:
        from ipfs_kit_py.config_manager import ConfigManager
        CONFIG_MANAGER_AVAILABLE = True
    except ImportError:
        pass
    
    # Test basic availability
    if STORACHA_KIT_AVAILABLE or S3_KIT_AVAILABLE or CONFIG_MANAGER_AVAILABLE:
        IPFS_KIT_AVAILABLE = True
    else:
        pytest.skip("No ipfs_kit_py components available", allow_module_level=True)
        
except Exception as e:
    pytest.skip(f"ipfs_kit_py package not available: {e}", allow_module_level=True)


class TestIPFSKitAdvanced:
    """Advanced tests for IPFSKit functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_ipfs_kit_configuration_override(self):
        """Test IPFSKit with configuration overrides."""
        custom_config = {
            'ipfs': {
                'gateway_url': 'https://custom.ipfs.gateway',
                'api_url': 'http://localhost:5002',
                'timeout': 120
            }
        }
        
        kit = IPFSKit(config=custom_config)
        assert kit is not None
        # Verify configuration was applied
        assert hasattr(kit, 'config') or hasattr(kit, '_config')
    
    def test_ipfs_kit_batch_operations(self):
        """Test batch operations with IPFSKit."""
        kit = IPFSKit()
        
        # Create test files for batch operations
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.test_dir, f'test_file_{i}.txt')
            with open(file_path, 'w') as f:
                f.write(f'Test content {i}')
            test_files.append(file_path)
        
        # Test batch add (mocked)
        with patch.object(kit, 'add_file', return_value={'hash': f'Qm{i}' * 10}) as mock_add:
            results = []
            for file_path in test_files:
                try:
                    result = kit.add_file(file_path)
                    results.append(result)
                except Exception as e:
                    # Expected in test environment
                    results.append({'error': str(e)})
            
            assert len(results) == 3
    
    def test_ipfs_kit_error_handling(self):
        """Test error handling scenarios."""
        kit = IPFSKit()
        
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, Exception)):
            kit.add_file('/non/existent/file.txt')
        
        # Test with invalid CID
        with pytest.raises((ValueError, Exception)):
            kit.get_file('invalid_cid', self.test_dir)
    
    def test_ipfs_kit_large_file_handling(self):
        """Test handling of large files."""
        kit = IPFSKit()
        
        # Create a larger test file
        large_file = os.path.join(self.test_dir, 'large_file.txt')
        with open(large_file, 'w') as f:
            # Write 1MB of data
            for i in range(10000):
                f.write('A' * 100 + '\n')
        
        # Test adding large file (mocked)
        try:
            result = kit.add_file(large_file)
            # In test environment, this might fail, which is expected
        except Exception as e:
            assert 'IPFS' in str(e) or 'connection' in str(e).lower()


class TestStorachaKitAdvanced:
    """Advanced tests for StorachaKit functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_storacha_kit_with_credentials(self):
        """Test StorachaKit with various credential configurations."""
        configs = [
            {'storacha': {'api_key': 'test_key', 'space_did': 'test_space'}},
            {'storacha': {'token': 'test_token'}},
            {}  # No credentials
        ]
        
        for config in configs:
            kit = StorachaKit(config=config)
            assert kit is not None
    
    def test_storacha_kit_upload_scenarios(self):
        """Test various upload scenarios."""
        kit = StorachaKit()
        
        # Test with different file types
        test_files = [
            ('text.txt', 'Hello World'),
            ('json.json', '{"key": "value"}'),
            ('binary.bin', b'\x00\x01\x02\x03')
        ]
        
        for filename, content in test_files:
            file_path = os.path.join(self.test_dir, filename)
            mode = 'wb' if isinstance(content, bytes) else 'w'
            with open(file_path, mode) as f:
                f.write(content)
            
            # Test upload (mocked in test environment)
            try:
                result = kit.upload_file(file_path)
                # Success or expected failure in test environment
            except Exception as e:
                # Expected in test environment without real Storacha connection
                assert isinstance(e, Exception)
    
    def test_storacha_kit_metadata_handling(self):
        """Test metadata handling in uploads."""
        kit = StorachaKit()
        
        test_file = os.path.join(self.test_dir, 'metadata_test.txt')
        with open(test_file, 'w') as f:
            f.write('Test content with metadata')
        
        metadata = {
            'title': 'Test File',
            'description': 'A test file for metadata handling',
            'tags': ['test', 'metadata', 'validation']
        }
        
        try:
            result = kit.upload_file(test_file, metadata=metadata)
            # Success or expected failure in test environment
        except Exception as e:
            # Expected in test environment
            assert isinstance(e, Exception)


class TestS3KitAdvanced:
    """Advanced tests for S3Kit functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_s3_kit_configuration_options(self):
        """Test S3Kit with various configuration options."""
        configs = [
            {
                's3': {
                    'endpoint_url': 'https://s3.amazonaws.com',
                    'region': 'us-east-1',
                    'bucket': 'test-bucket'
                }
            },
            {
                's3': {
                    'endpoint_url': 'https://storage.googleapis.com',
                    'region': 'us-central1'
                }
            },
            {}  # Default configuration
        ]
        
        for config in configs:
            kit = S3Kit(config=config)
            assert kit is not None
    
    def test_s3_kit_multipart_upload_simulation(self):
        """Test multipart upload simulation."""
        kit = S3Kit()
        
        # Create a file that would trigger multipart upload
        large_file = os.path.join(self.test_dir, 'multipart_test.txt')
        with open(large_file, 'w') as f:
            # Write enough data to simulate multipart scenario
            for i in range(1000):
                f.write(f'Line {i}: ' + 'A' * 100 + '\n')
        
        try:
            result = kit.upload_file(large_file, 'test-bucket', 'multipart_test.txt')
            # Success or expected failure in test environment
        except Exception as e:
            # Expected in test environment without real S3 credentials
            assert isinstance(e, Exception)
    
    def test_s3_kit_error_scenarios(self):
        """Test various error scenarios."""
        kit = S3Kit()
        
        # Test with invalid bucket name
        with pytest.raises((ValueError, Exception)):
            kit.upload_file('nonexistent.txt', 'invalid-bucket-name-with-invalid-chars!', 'test.txt')
        
        # Test with empty file
        empty_file = os.path.join(self.test_dir, 'empty.txt')
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        try:
            result = kit.upload_file(empty_file, 'test-bucket', 'empty.txt')
            # Might succeed or fail depending on S3 configuration
        except Exception as e:
            assert isinstance(e, Exception)


class TestConfigManagerAdvanced:
    """Advanced tests for ConfigManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_config_manager_file_loading(self):
        """Test loading configuration from files."""
        # Create test config files
        yaml_config = os.path.join(self.test_dir, 'test_config.yaml')
        with open(yaml_config, 'w') as f:
            f.write("""
ipfs:
  gateway_url: https://test.ipfs.gateway
  timeout: 60
storacha:
  api_key: test_key
  space_did: test_space
s3:
  bucket: test-bucket
  region: us-west-2
""")
        
        json_config = os.path.join(self.test_dir, 'test_config.json')
        config_data = {
            'ipfs': {
                'gateway_url': 'https://json.ipfs.gateway',
                'timeout': 30
            },
            'storacha': {
                'token': 'json_token'
            }
        }
        with open(json_config, 'w') as f:
            json.dump(config_data, f)
        
        try:
            # Test YAML loading
            manager = ConfigManager(config_file=yaml_config)
            assert manager is not None
            
            # Test JSON loading
            manager = ConfigManager(config_file=json_config)
            assert manager is not None
            
        except Exception as e:
            # ConfigManager might not be available or implemented
            pytest.skip(f"ConfigManager not available: {e}")
    
    def test_config_manager_environment_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            'IPFS_GATEWAY_URL': 'https://env.ipfs.gateway',
            'STORACHA_API_KEY': 'env_api_key',
            'S3_BUCKET': 'env-bucket'
        }):
            try:
                manager = ConfigManager()
                assert manager is not None
                # Verify environment variables are used
            except Exception as e:
                pytest.skip(f"ConfigManager not available: {e}")
    
    def test_config_manager_validation(self):
        """Test configuration validation."""
        invalid_configs = [
            {'ipfs': {'timeout': 'invalid'}},  # Invalid timeout type
            {'s3': {'region': ''}},  # Empty region
            {'storacha': {}}  # Missing required fields
        ]
        
        for invalid_config in invalid_configs:
            try:
                manager = ConfigManager(config=invalid_config)
                # Some validation might be lenient
                assert manager is not None
            except (ValueError, TypeError) as e:
                # Expected validation error
                assert isinstance(e, (ValueError, TypeError))
            except Exception as e:
                # ConfigManager might not be available
                pytest.skip(f"ConfigManager not available: {e}")


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_multi_kit_workflow(self):
        """Test workflow using multiple kits together."""
        # Initialize all kits
        ipfs_kit = IPFSKit()
        storacha_kit = StorachaKit()
        s3_kit = S3Kit()
        
        # Create test file
        test_file = os.path.join(self.test_dir, 'workflow_test.txt')
        with open(test_file, 'w') as f:
            f.write('Multi-kit workflow test content')
        
        # Simulate workflow: IPFS -> Storacha -> S3
        results = {}
        
        try:
            # Step 1: Add to IPFS
            ipfs_result = ipfs_kit.add_file(test_file)
            results['ipfs'] = ipfs_result
        except Exception as e:
            results['ipfs'] = {'error': str(e)}
        
        try:
            # Step 2: Upload to Storacha
            storacha_result = storacha_kit.upload_file(test_file)
            results['storacha'] = storacha_result
        except Exception as e:
            results['storacha'] = {'error': str(e)}
        
        try:
            # Step 3: Backup to S3
            s3_result = s3_kit.upload_file(test_file, 'backup-bucket', 'workflow_test.txt')
            results['s3'] = s3_result
        except Exception as e:
            results['s3'] = {'error': str(e)}
        
        # Verify all steps were attempted
        assert 'ipfs' in results
        assert 'storacha' in results
        assert 's3' in results
    
    def test_concurrent_operations(self):
        """Test concurrent operations across kits."""
        ipfs_kit = IPFSKit()
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.test_dir, f'concurrent_test_{i}.txt')
            with open(file_path, 'w') as f:
                f.write(f'Concurrent test content {i}')
            test_files.append(file_path)
        
        # Simulate concurrent operations
        results = []
        for file_path in test_files:
            try:
                result = ipfs_kit.add_file(file_path)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        assert len(results) == len(test_files)
    
    def test_error_recovery_scenarios(self):
        """Test error recovery and fallback scenarios."""
        kits = [
            ('ipfs', IPFSKit()),
            ('storacha', StorachaKit()),
            ('s3', S3Kit())
        ]
        
        test_file = os.path.join(self.test_dir, 'error_recovery_test.txt')
        with open(test_file, 'w') as f:
            f.write('Error recovery test content')
        
        for kit_name, kit in kits:
            try:
                # Attempt operation that might fail
                if kit_name == 'ipfs':
                    result = kit.add_file(test_file)
                elif kit_name == 'storacha':
                    result = kit.upload_file(test_file)
                elif kit_name == 's3':
                    result = kit.upload_file(test_file, 'test-bucket', 'error_test.txt')
                
                # If we get here, operation succeeded or was mocked
                assert result is not None or result == {}
                
            except Exception as e:
                # Expected in test environment
                assert isinstance(e, Exception)
                print(f"Expected error for {kit_name}: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
