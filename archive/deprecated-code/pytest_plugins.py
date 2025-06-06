"""
Pytest plugin to mock problematic modules before they can be imported
"""
import sys
from unittest.mock import MagicMock

def pytest_configure():
    """Configure pytest by setting up early mocks"""
    # Mock IPFS-related modules to prevent installation during tests
    sys.modules['ipfs_kit_py.ipfs_kit'] = MagicMock()
    sys.modules['ipfs_kit_py.install_ipfs'] = MagicMock()
    # DEPRECATED: storacha_clusters.storacha_clusters is deprecated
    sys.modules['storacha_clusters.storacha_clusters'] = MagicMock()
    # Add ipfs_kit_py mocks for the new implementation
    sys.modules['ipfs_kit_py.storacha_kit'] = MagicMock()
    
    # Mock the ipfs_kit function specifically
    mock_ipfs_kit = MagicMock()
    mock_install_ipfs = MagicMock()
    
    # Create a mock install_ipfs that returns a mock with install_ipfs_daemon method
    mock_installer = MagicMock()
    mock_installer.install_ipfs_daemon.return_value = None
    mock_install_ipfs.return_value = mock_installer
    
    sys.modules['ipfs_kit_py'].ipfs_kit = mock_ipfs_kit
    sys.modules['ipfs_kit_py'].install_ipfs = mock_install_ipfs
