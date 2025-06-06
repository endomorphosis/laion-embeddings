#!/usr/bin/env python3
"""
Quick status check for ipfs_kit_py integration
"""

import sys
import os

# Add ipfs_kit_py to path
sys.path.insert(0, 'docs/ipfs_kit_py')

def check_component(component_name, import_func):
    """Check if a component is working"""
    try:
        result = import_func()
        print(f"✅ {component_name}: Working")
        return True, result
    except Exception as e:
        print(f"❌ {component_name}: {str(e)[:100]}...")
        return False, str(e)

def test_s3_kit():
    from ipfs_kit_py.s3_kit import s3_kit
    kit = s3_kit(resources={})
    methods = [m for m in dir(kit) if not m.startswith('_') and callable(getattr(kit, m))]
    return f"Instantiated with {len(methods)} methods"

def test_api_stability():
    from ipfs_kit_py.api_stability import APIStability, API_REGISTRY
    return f"API stability system available with {len(API_REGISTRY)} registries"

def test_arc_cache():
    from ipfs_kit_py.arc_cache import ARCache
    cache = ARCache(maxsize=1024)  # Use maxsize instead of capacity
    cache.put("test", "value")
    return f"Cache works: {cache.get('test')}"

def test_storacha_exceptions():
    from ipfs_kit_py.storacha_kit import IPFSError, IPFSConnectionError
    return f"Exception classes available: IPFSError, IPFSConnectionError"

def main():
    print("=== ipfs_kit_py Status Check ===\n")
    
    # Test core components
    components = [
        ("s3_kit", test_s3_kit),
        ("api_stability", test_api_stability), 
        ("arc_cache", test_arc_cache),
        ("storacha_exceptions", test_storacha_exceptions)
    ]
    
    working_count = 0
    total_count = len(components)
    
    for name, test_func in components:
        success, result = check_component(name, test_func)
        if success:
            working_count += 1
            print(f"   └─ {result}")
        print()
    
    print(f"=== Summary ===")
    print(f"Working components: {working_count}/{total_count}")
    print(f"Success rate: {(working_count/total_count)*100:.1f}%")
    
    if working_count >= 3:
        print("\n🎉 Migration is SUCCESSFUL!")
        print("Core functionality is working. Optional dependencies can be installed for full features.")
    else:
        print("\n⚠️  Some components need attention.")

if __name__ == "__main__":
    main()
