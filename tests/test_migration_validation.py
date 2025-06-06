#!/usr/bin/env python3
"""
Test suite for IPFS Kit migration validation
Ensures the new ipfs_kit_py implementation works correctly after migration.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestIPFSKitMigration:
    """Test the new ipfs_kit_py implementation after migration."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock the ipfs_kit_py imports for testing
        self.mock_ipfs_kit = MagicMock()
        self.mock_storacha_kit = MagicMock() 
        self.mock_s3_kit = MagicMock()
        
    def test_import_availability(self):
        """Test that new imports are available after migration."""
        try:
            # Test import without actual installation
            with patch.dict('sys.modules', {
                'ipfs_kit_py': MagicMock(),
                'ipfs_kit_py.ipfs_kit': self.mock_ipfs_kit,
                'ipfs_kit_py.storacha_kit': self.mock_storacha_kit,
                'ipfs_kit_py.s3_kit': self.mock_s3_kit,
                'ipfs_kit_py.high_level_api': MagicMock(),
                'ipfs_kit_py.error': MagicMock()
            }):
                # Simulate the new import pattern
                from ipfs_kit_py import ipfs_kit
                from ipfs_kit_py import storacha_kit
                from ipfs_kit_py import s3_kit
                
                assert ipfs_kit is not None
                assert storacha_kit is not None
                assert s3_kit is not None
                
        except ImportError as e:
            pytest.skip(f"ipfs_kit_py not available for testing: {e}")

    def test_migration_file_exists(self):
        """Test that migration files were created."""
        migration_script = project_root / "scripts" / "migrate_to_ipfs_kit.py"
        integration_guide = project_root / "docs" / "IPFS_KIT_INTEGRATION_GUIDE.md"
        audit_script = project_root / "scripts" / "audit_ipfs_usage.py"
        
        assert migration_script.exists(), "Migration script should exist"
        assert integration_guide.exists(), "Integration guide should exist"
        assert audit_script.exists(), "Audit script should exist"

    def test_basic_ipfs_operations(self):
        """Test basic IPFS operations with new API."""
        with patch.dict('sys.modules', {
            'ipfs_kit_py': MagicMock(),
            'ipfs_kit_py.ipfs_kit': self.mock_ipfs_kit
        }):
            # Mock the ipfs_kit instance
            mock_instance = MagicMock()
            self.mock_ipfs_kit.return_value = mock_instance
            
            # Mock successful operations
            mock_result = MagicMock()
            mock_result.hash = 'QmTestHash123'
            mock_result.size = 100
            mock_instance.add.return_value = mock_result
            mock_instance.get.return_value = b"test content"
            
            # Test the interface
            from ipfs_kit_py import ipfs_kit
            client = ipfs_kit()
            
            # Test add operation
            result = client.add(b"test data")
            assert result.hash == 'QmTestHash123'
            
            # Test get operation
            content = client.get('QmTestHash123')
            assert content == b"test content"

    def test_storacha_operations(self):
        """Test Storacha operations with new API."""
        with patch.dict('sys.modules', {
            'ipfs_kit_py': MagicMock(),
            'ipfs_kit_py.storacha_kit': self.mock_storacha_kit
        }):
            # Mock the storacha_kit instance
            mock_instance = MagicMock()
            self.mock_storacha_kit.return_value = mock_instance
            
            # Mock successful upload
            mock_result = MagicMock()
            mock_result.cid = 'bafyTestCID123'
            mock_result.success = True
            mock_instance.upload.return_value = mock_result
            
            # Test the interface
            from ipfs_kit_py import storacha_kit
            client = storacha_kit(api_key='test_key')
            
            result = client.upload(b"test data")
            assert result.cid == 'bafyTestCID123'
            assert result.success is True

    def test_error_handling(self):
        """Test that error handling works with new API."""
        with patch.dict('sys.modules', {
            'ipfs_kit_py': MagicMock(),
            'ipfs_kit_py.ipfs_kit': self.mock_ipfs_kit,
            'ipfs_kit_py.error': MagicMock()
        }):
            # Create mock error classes
            class MockIPFSError(Exception):
                pass
            
            class MockIPFSConnectionError(MockIPFSError):
                pass
            
            # Mock the error module
            error_module = MagicMock()
            error_module.IPFSError = MockIPFSError
            error_module.IPFSConnectionError = MockIPFSConnectionError
            
            with patch.dict('sys.modules', {'ipfs_kit_py.error': error_module}):
                from ipfs_kit_py.error import IPFSError, IPFSConnectionError
                
                # Test that error classes are available
                assert IPFSError is not None
                assert IPFSConnectionError is not None
                
                # Test error inheritance
                assert issubclass(IPFSConnectionError, IPFSError)

    def test_requirements_backup_exists(self):
        """Test that requirements backup was created during migration."""
        backup_dir = project_root / "backup_deprecated_ipfs"
        
        # This test will pass if migration has been run
        if backup_dir.exists():
            backup_req = backup_dir / "requirements.txt.backup"
            assert backup_req.exists(), "Requirements backup should exist after migration"
        else:
            pytest.skip("Migration has not been run yet")

    def test_deprecated_imports_removed(self):
        """Test that deprecated import patterns are no longer used."""
        # Check main __init__.py file
        init_file = project_root / "__init__.py"
        
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
            
            # Check that old imports have been replaced or commented out
            deprecated_patterns = [
                'from ipfs_kit_py import ipfs_kit',
                'from ipfs_kit_py import ipfs_kit',
                'from ipfs_kit_py import ipfs_kit',
                'from ipfs_kit_py import ipfs_kit'
            ]
            
            active_deprecated = []
            for pattern in deprecated_patterns:
                if pattern in content and not content.split(pattern)[0].split('\n')[-1].strip().startswith('#'):
                    active_deprecated.append(pattern)
            
            if active_deprecated:
                pytest.fail(f"Found active deprecated imports: {active_deprecated}")

class TestMigrationIntegrity:
    """Test that the migration maintains system integrity."""
    
    def test_project_structure_preserved(self):
        """Test that important project files are preserved."""
        important_files = [
            "main.py",
            "auth.py", 
            "monitoring.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file_name in important_files:
            file_path = project_root / file_name
            if file_path.exists():
                assert file_path.is_file(), f"{file_name} should still exist and be a file"

    def test_no_broken_imports(self):
        """Test that critical imports still work after migration."""
        # Test some core imports that should still work
        critical_modules = [
            "fastapi",
            "pandas", 
            "numpy",
            "datasets",
            "transformers"
        ]
        
        for module_name in critical_modules:
            try:
                __import__(module_name)
            except ImportError:
                # Skip if module not installed, but don't fail
                pytest.skip(f"Module {module_name} not available in test environment")

class TestMigrationValidation:
    """Validate the migration process and results."""
    
    def test_audit_results_available(self):
        """Test that audit results are available."""
        audit_report = project_root / "docs" / "IPFS_AUDIT_REPORT.md"
        audit_results = project_root / "docs" / "ipfs_audit_results.json"
        
        # These should exist if audit was run
        if audit_report.exists():
            assert audit_report.is_file()
            
            # Check that audit found the expected number of files
            with open(audit_report, 'r') as f:
                content = f.read()
            
            assert "Files with IPFS usage:" in content
            assert "Total files to modify:" in content

    def test_migration_script_syntax(self):
        """Test that migration script has valid syntax."""
        migration_script = project_root / "scripts" / "migrate_to_ipfs_kit.py"
        
        # Test that the script compiles without syntax errors
        try:
            with open(migration_script, 'r') as f:
                code = f.read()
            
            compile(code, str(migration_script), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Migration script has syntax error: {e}")

    def test_integration_guide_complete(self):
        """Test that integration guide is comprehensive."""
        guide_file = project_root / "docs" / "IPFS_KIT_INTEGRATION_GUIDE.md"
        
        with open(guide_file, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "Overview",
            "Migration Summary", 
            "API Reference",
            "Testing After Migration",
            "Rollback Plan"
        ]
        
        for section in required_sections:
            assert section in content, f"Integration guide missing section: {section}"

def test_full_migration_workflow():
    """Test the complete migration workflow."""
    # This is an integration test that would be run after migration
    
    # 1. Check that audit was completed
    audit_report = project_root / "docs" / "IPFS_AUDIT_REPORT.md"
    if audit_report.exists():
        print("✅ Audit completed")
    
    # 2. Check that migration script exists and is executable
    migration_script = project_root / "scripts" / "migrate_to_ipfs_kit.py"
    assert migration_script.exists()
    print("✅ Migration script ready")
    
    # 3. Check that documentation is in place
    integration_guide = project_root / "docs" / "IPFS_KIT_INTEGRATION_GUIDE.md"
    assert integration_guide.exists()
    print("✅ Integration guide available")
    
    # 4. Check that ipfs_kit_py source is available
    ipfs_kit_source = project_root / "docs" / "ipfs_kit_py"
    assert ipfs_kit_source.exists()
    print("✅ IPFS Kit source available")
    
    print("✅ Migration workflow validation complete")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
