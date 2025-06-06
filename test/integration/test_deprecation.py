#!/usr/bin/env python3
"""
Test script to verify storacha_clusters deprecation.
"""

import sys
import warnings
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_deprecation():
    """Test that storacha_clusters shows deprecation warnings."""
    print("Testing storacha_clusters deprecation...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import the deprecated module
        try:
            from storacha_clusters import storacha_clusters
            print("✓ Import successful")
        except Exception as e:
            print(f"✗ Import failed: {e}")
            return False
        
        # Test instantiation
        try:
            sc = storacha_clusters(resources={}, metadata={})
            print("✓ Instantiation successful")
        except DeprecationWarning as e:
            print(f"✓ DeprecationWarning on instantiation: {e}")
        except Exception as e:
            print(f"✗ Unexpected error on instantiation: {e}")
            return False
        
        # Test method calls
        try:
            result = sc.test()
            print("✗ test() method should have raised DeprecationWarning")
            return False
        except DeprecationWarning as e:
            print(f"✓ test() method raised DeprecationWarning: {e}")
        except Exception as e:
            print(f"✗ Unexpected error calling test(): {e}")
            return False
        
        # Check warnings
        print(f"\nWarnings captured: {len(w)}")
        for i, warning in enumerate(w):
            print(f"  {i+1}. {warning.category.__name__}: {warning.message}")
    
    print("\n✓ Deprecation test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_deprecation()
    sys.exit(0 if success else 1)
