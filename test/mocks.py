"""
Mock implementations for external dependencies.

This module provides mock implementations of various external dependencies
to avoid circular imports during testing.
"""

import sys
from unittest.mock import MagicMock

class MockModule:
    """A mock module that can be used to replace problematic imports."""
    
    def __init__(self, name):
        self.name = name
        self._items = {}  # For making the module behave like a dictionary
        
    def __getattr__(self, attr):
        # Return a MockModule for submodules or a MagicMock for attributes
        return MockModule(f"{self.name}.{attr}")
    
    def __call__(self, *args, **kwargs):
        # Make the mock callable
        return MagicMock()
        
    # Make the module iterable to handle `from X import Y` statements
    def __iter__(self):
        # Return an empty iterator to avoid errors
        return iter([])
    
    # Dict-like behavior for modules
    def __getitem__(self, key):
        if key not in self._items:
            self._items[key] = MockModule(f"{self.name}.{key}")
        return self._items[key]
    
    def get(self, key, default=None):
        return self._items.get(key, default)
        
    def items(self):
        return self._items.items()
        
    def keys(self):
        return self._items.keys()
        
    def values(self):
        return self._items.values()

def patch_modules():
    """Patch problematic modules with mock implementations."""
    # First handle the transformers.utils issue
    try:
        import transformers
        if not hasattr(transformers, 'utils'):
            # Create the utils module if it doesn't exist
            utils_mock = MockModule('transformers.utils')
            utils_mock.PushToHubMixin = MagicMock()
            transformers.utils = utils_mock
            sys.modules['transformers.utils'] = utils_mock
    except ImportError:
        pass
    
    # Add only truly problematic modules that should always be mocked
    # DO NOT include sklearn here - it should be available when installed
    modules_to_patch = [
        'ipfs_accelerate_py',
        'ipfs_datasets_py',
        'ipfs_kit.rag_query_optimizer',
        'ipfs_kit.worker',
    ]
    
    # Special handling for torchvision - mock it without trying to import
    if 'torchvision' not in sys.modules:
        sys.modules['torchvision'] = MockModule('torchvision')
        # Add submodules
        sys.modules['torchvision.transforms'] = MockModule('torchvision.transforms')
    
    # Only patch modules that are not already imported or don't exist
    for module_name in modules_to_patch:
        # Skip trying to import these modules - just mock them directly
        if module_name not in sys.modules:
            parts = module_name.split('.')
            parent = '.'.join(parts[:-1])
            
            if parent and parent not in sys.modules:
                # If parent module doesn't exist, create it
                sys.modules[parent] = MockModule(parent)
                
            # Create the module with our mock
            sys.modules[module_name] = MockModule(module_name)
        
    # Special handling for torch.library which causes issues
    if 'torch' in sys.modules:
        torch_mock = sys.modules['torch']
        torch_mock.library = MagicMock()
    
    # Special handling for ipfs_kit.rag_query_optimizer
    if 'ipfs_kit.rag_query_optimizer' in sys.modules:
        optimizer_mock = sys.modules['ipfs_kit.rag_query_optimizer']
        # Add the missing VectorIndexPartitioner class
        class VectorIndexPartitioner(MagicMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        
        optimizer_mock.VectorIndexPartitioner = VectorIndexPartitioner
        
def setup_test_environment():
    """Set up the test environment with necessary mocks and configurations."""
    # Set testing environment variable
    import os
    os.environ['TESTING'] = 'true'
    
    # Patch problematic modules
    patch_modules()

# Auto-initialize when imported
setup_test_environment()
