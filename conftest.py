import pytest
import sys
from unittest.mock import MagicMock, patch

# Prevent IPFS installation by mocking modules before any imports
sys.modules['ipfs_kit_py.storacha_kit'] = MagicMock()  
sys.modules['ipfs_kit_py.s3_kit'] = MagicMock()
sys.modules['ipfs_kit_py.high_level_api'] = MagicMock()

sys.modules['ipfs_kit_py.install_ipfs'] = MagicMock()
sys.modules['storacha_clusters.storacha_clusters'] = MagicMock()

# Removed problematic mock for ipfs_kit_py.ipfs_kit
# @pytest.fixture(autouse=True)
# def mock_ipfs_kit():
#     """
#     Fixture to mock ipfs_kit_py.ipfs_kit and prevent IPFS installation during tests.
#     """
#     with patch('ipfs_kit_py.ipfs_kit', new=MagicMock()) as mock_kit:
#         yield mock_kit

@pytest.fixture(autouse=True)
def mock_install_ipfs():
    """
    Fixture to mock ipfs_kit_py.install_ipfs and prevent IPFS installation during tests.
    """
    try:
        with patch('ipfs_kit_py.install_ipfs.install_ipfs', new=MagicMock()) as mock_install:
            yield mock_install
    except ImportError:
        # If module structure is different, try alternative paths
        try:
            with patch('ipfs_kit_py.install_ipfs', new=MagicMock()) as mock_install:
                yield mock_install
        except ImportError:
            # Skip mocking if module not available
            yield None

@pytest.fixture(autouse=True)
def mock_storacha_clusters():
    """
    Fixture to mock storacha_clusters and prevent IPFS installation during tests.
    """
    with patch('storacha_clusters.storacha_clusters', new=MagicMock()) as mock_storacha:
        yield mock_storacha
