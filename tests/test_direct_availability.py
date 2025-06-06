#!/usr/bin/env python3
"""
Direct test of ipfs_kit_py component availability.
"""

import sys
import os
import tempfile

# Add the package path
sys.path.insert(0, '/home/barberb/laion-embeddings-1/docs/ipfs_kit_py')

def test_component_availability():
    """Test what components are available."""
    results = []
    
    # Test individual module imports
    modules_to_test = [
        'ipfs_kit_py.storacha_kit',
        'ipfs_kit_py.s3_kit', 
        'ipfs_kit_py.config_manager',
        'ipfs_kit_py.api_stability',
        'ipfs_kit_py.arc_cache'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            results.append((module_name, 'success', 'Available'))
            print(f"✓ {module_name}: Available")
        except ImportError as e:
            results.append((module_name, 'import_error', str(e)))
            print(f"✗ {module_name}: ImportError - {e}")
        except Exception as e:
            results.append((module_name, 'error', str(e)))
            print(f"? {module_name}: Error - {e}")
    
    return results

def test_class_instantiation():
    """Test class instantiation for available modules."""
    results = []
    
    # Test StorachaKit
    try:
        from ipfs_kit_py.storacha_kit import StorachaKit
        kit = StorachaKit()
        results.append(('StorachaKit', 'success', 'Instantiated'))
        print("✓ StorachaKit: Instantiated successfully")
    except ImportError as e:
        results.append(('StorachaKit', 'import_error', str(e)))
        print(f"✗ StorachaKit: ImportError - {e}")
    except Exception as e:
        results.append(('StorachaKit', 'runtime_error', str(e)))
        print(f"? StorachaKit: Runtime error - {e}")
    
    # Test S3Kit
    try:
        from ipfs_kit_py.s3_kit import S3Kit
        kit = S3Kit()
        results.append(('S3Kit', 'success', 'Instantiated'))
        print("✓ S3Kit: Instantiated successfully")
    except ImportError as e:
        results.append(('S3Kit', 'import_error', str(e)))
        print(f"✗ S3Kit: ImportError - {e}")
    except Exception as e:
        results.append(('S3Kit', 'runtime_error', str(e)))
        print(f"? S3Kit: Runtime error - {e}")
    
    # Test ConfigManager
    try:
        from ipfs_kit_py.config_manager import ConfigManager
        manager = ConfigManager()
        results.append(('ConfigManager', 'success', 'Instantiated'))
        print("✓ ConfigManager: Instantiated successfully")
    except ImportError as e:
        results.append(('ConfigManager', 'import_error', str(e)))
        print(f"✗ ConfigManager: ImportError - {e}")
    except Exception as e:
        results.append(('ConfigManager', 'runtime_error', str(e)))
        print(f"? ConfigManager: Runtime error - {e}")
    
    return results

def test_file_operations():
    """Test basic file operations with available kits."""
    results = []
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('Test content for ipfs_kit_py integration')
        test_file = f.name
    
    try:
        # Test StorachaKit file operations
        try:
            from ipfs_kit_py.storacha_kit import StorachaKit
            kit = StorachaKit()
            
            # Test upload (will likely fail without credentials, but we can test the interface)
            try:
                result = kit.upload_file(test_file)
                results.append(('StorachaKit.upload_file', 'success', result))
                print("✓ StorachaKit.upload_file: Success")
            except Exception as e:
                results.append(('StorachaKit.upload_file', 'expected_error', str(e)))
                print(f"? StorachaKit.upload_file: Expected error - {e}")
                
        except ImportError as e:
            results.append(('StorachaKit', 'import_error', str(e)))
        
        # Test S3Kit file operations
        try:
            from ipfs_kit_py.s3_kit import S3Kit
            kit = S3Kit()
            
            try:
                result = kit.upload_file(test_file, 'test-bucket', 'test-key')
                results.append(('S3Kit.upload_file', 'success', result))
                print("✓ S3Kit.upload_file: Success")
            except Exception as e:
                results.append(('S3Kit.upload_file', 'expected_error', str(e)))
                print(f"? S3Kit.upload_file: Expected error - {e}")
                
        except ImportError as e:
            results.append(('S3Kit', 'import_error', str(e)))
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
    
    return results

if __name__ == '__main__':
    print("=== Testing ipfs_kit_py Component Availability ===\n")
    
    print("1. Module Import Tests:")
    import_results = test_component_availability()
    
    print("\n2. Class Instantiation Tests:")
    instantiation_results = test_class_instantiation()
    
    print("\n3. File Operation Tests:")
    operation_results = test_file_operations()
    
    print("\n=== Summary ===")
    total_tests = len(import_results) + len(instantiation_results) + len(operation_results)
    successful_tests = sum(1 for _, status, _ in import_results + instantiation_results + operation_results if status == 'success')
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Issues: {total_tests - successful_tests}")
    
    if successful_tests > 0:
        print("\n✓ Some ipfs_kit_py functionality is available for integration")
    else:
        print("\n✗ No ipfs_kit_py functionality currently available")
    
    print("\nComponent readiness for laion-embeddings-1 integration:")
    available_components = [name for name, status, _ in instantiation_results if status == 'success']
    if available_components:
        print(f"✓ Available: {', '.join(available_components)}")
    else:
        print("? No components fully available (may need configuration or dependencies)")
