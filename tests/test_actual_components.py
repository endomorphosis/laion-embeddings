#!/usr/bin/env python3
"""
Corrected test for ipfs_kit_py components with actual class names.
"""

import sys
import os
import tempfile

# Add the package path
sys.path.insert(0, '/home/barberb/laion-embeddings-1/docs/ipfs_kit_py')

def test_actual_components():
    """Test components using their actual class names."""
    results = []
    
    # Test s3_kit (lowercase)
    try:
        from ipfs_kit_py.s3_kit import s3_kit
        kit = s3_kit(resources={})
        results.append(('s3_kit', 'success', 'Instantiated'))
        print("✓ s3_kit: Instantiated successfully")
        
        # Test some methods exist
        methods = ['s3_cp_file', 's3_ls_dir', 's3_ul_file']
        for method in methods:
            if hasattr(kit, method):
                print(f"  ✓ Method {method} available")
            else:
                print(f"  ? Method {method} not found")
                
    except Exception as e:
        results.append(('s3_kit', 'error', str(e)))
        print(f"✗ s3_kit: Error - {e}")
    
    # Test storacha_kit classes
    try:
        from ipfs_kit_py.storacha_kit import IPFSError, IPFSConnectionError
        print("✓ storacha_kit: Exception classes imported")
        results.append(('storacha_kit_exceptions', 'success', 'Available'))
        
        # Look for actual kit classes
        import ipfs_kit_py.storacha_kit as storacha_module
        
        # Get all classes from the module
        classes = [name for name in dir(storacha_module) if isinstance(getattr(storacha_module, name), type)]
        print(f"  Available classes: {classes}")
        
    except Exception as e:
        results.append(('storacha_kit', 'error', str(e)))
        print(f"✗ storacha_kit: Error - {e}")
    
    # Test api_stability
    try:
        from ipfs_kit_py.api_stability import stable_api, beta_api, experimental_api
        print("✓ api_stability: Decorators imported")
        results.append(('api_stability', 'success', 'Available'))
    except Exception as e:
        results.append(('api_stability', 'error', str(e)))
        print(f"✗ api_stability: Error - {e}")
    
    return results

def test_file_operations():
    """Test actual file operations."""
    results = []
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('Test content for integration')
        test_file = f.name
    
    try:
        # Test s3_kit operations
        try:
            from ipfs_kit_py.s3_kit import s3_kit
            kit = s3_kit(resources={})
            
            # Test method calls (will likely fail without proper config)
            if hasattr(kit, 's3_ul_file'):
                try:
                    # This will likely fail, but we test the interface
                    kit.s3_ul_file(test_file, 'test-bucket', 'test-key')
                    results.append(('s3_kit.s3_ul_file', 'success', 'Worked'))
                except Exception as e:
                    results.append(('s3_kit.s3_ul_file', 'expected_error', str(e)))
                    print(f"? s3_kit.s3_ul_file: Expected error - {e}")
            
        except Exception as e:
            results.append(('s3_kit', 'error', str(e)))
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
    
    return results

def check_imports_and_structure():
    """Check what's actually importable from the package."""
    
    # Check main package
    try:
        import ipfs_kit_py
        print("✓ Main package imports successfully")
        
        # Check what's available in the package
        if hasattr(ipfs_kit_py, '__all__'):
            print(f"  Exports: {ipfs_kit_py.__all__}")
        
        # List available attributes
        attrs = [attr for attr in dir(ipfs_kit_py) if not attr.startswith('_')]
        print(f"  Available attributes: {attrs[:10]}...")  # First 10
        
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
    
    # Check individual modules that seem to work
    working_modules = []
    
    modules_to_check = [
        'ipfs_kit_py.s3_kit',
        'ipfs_kit_py.api_stability', 
        'ipfs_kit_py.arc_cache'
    ]
    
    for module_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[''])
            working_modules.append(module_name)
            print(f"✓ {module_name}: Imports successfully")
            
            # List classes and functions
            items = [name for name in dir(module) if not name.startswith('_') and isinstance(getattr(module, name), (type, type(lambda: None)))]
            if items:
                print(f"  Contains: {items[:5]}...")  # First 5 items
                
        except Exception as e:
            print(f"✗ {module_name}: {e}")
    
    return working_modules

if __name__ == '__main__':
    print("=== Testing ipfs_kit_py Actual Components ===\n")
    
    print("1. Import and Structure Check:")
    working_modules = check_imports_and_structure()
    
    print("\n2. Component Tests:")
    component_results = test_actual_components()
    
    print("\n3. File Operation Tests:")
    operation_results = test_file_operations()
    
    print("\n=== Final Assessment ===")
    
    successful_components = [name for name, status, _ in component_results if status == 'success']
    working_operations = [name for name, status, _ in operation_results if status in ['success', 'expected_error']]
    
    print(f"Working modules: {len(working_modules)}")
    print(f"Successful components: {len(successful_components)}")
    print(f"Tested operations: {len(working_operations)}")
    
    if working_modules and successful_components:
        print("\n✓ ipfs_kit_py has some functional components")
        print("✓ Integration with laion-embeddings-1 is partially feasible")
        print("\nRecommendations:")
        print("- Use s3_kit for S3 operations")
        print("- Use api_stability decorators")
        print("- Handle missing dependencies gracefully")
        print("- Consider fixing WebSocket dependencies for full functionality")
    else:
        print("\n? ipfs_kit_py has significant issues")
        print("? Full integration may require dependency fixes")
