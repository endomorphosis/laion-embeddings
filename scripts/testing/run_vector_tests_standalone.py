#!/usr/bin/env python3
"""
Standalone runner for vector service tests
"""
import os
import sys
import asyncio
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set testing environment
os.environ['TESTING'] = 'true'

# Mock problematic modules early
sys.modules['ipfs_kit_py.ipfs_kit'] = MagicMock()
sys.modules['ipfs_kit_py.install_ipfs'] = MagicMock()
sys.modules['storacha_clusters.storacha_clusters'] = MagicMock()

def run_vector_service_tests():
    """Run vector service tests without pytest"""
    
    print("=" * 60)
    print("VECTOR SERVICE TESTS")
    print("=" * 60)
    
    try:
        # Import test modules
        from test.test_vector_service import (
            TestVectorConfig, TestFAISSIndex, TestVectorService,
            TestVectorServiceAsync, TestVectorServiceIntegration
        )
        
        # Run VectorConfig tests
        print("\n1. Testing VectorConfig")
        config_test = TestVectorConfig()
        config_test.test_vector_config_defaults()
        config_test.test_vector_config_custom()
        config_test.test_vector_config_testing_mode()
        print("   ✓ VectorConfig tests passed (3/3)")
        
        # Run FAISSIndex tests (need to create fixtures)
        print("\n2. Testing FAISSIndex")
        from services.vector_service import VectorConfig
        import numpy as np
        
        vector_config = VectorConfig(dimension=128, index_type="Flat")
        test_vectors = np.array([[0.1, 0.2] * 64, [0.3, 0.4] * 64, [0.5, 0.6] * 64], dtype=np.float32)
        
        index_test = TestFAISSIndex()
        index_test.test_faiss_index_initialization(vector_config)
        index_test.test_flat_index_creation()
        index_test.test_ivf_index_creation()
        index_test.test_add_vectors_flat(vector_config, test_vectors)
        index_test.test_search_vectors(vector_config, test_vectors)
        index_test.test_train_ivf_index()
        
        # Create temporary directory for save/load test
        import tempfile
        temp_dir = tempfile.mkdtemp()
        index_test.test_save_and_load_index(vector_config, test_vectors, temp_dir)
        
        index_test.test_normalize_vectors()
        print("   ✓ FAISSIndex tests passed (8/8)")
        
        # Run VectorService tests
        print("\n3. Testing VectorService")
        from services.vector_service import VectorService
        
        vector_config = VectorConfig(dimension=128, index_type="Flat")
        test_vectors = np.array([[0.1, 0.2] * 64, [0.3, 0.4] * 64, [0.5, 0.6] * 64], dtype=np.float32)
        test_metadata = [{"id": i, "text": f"Text {i}"} for i in range(3)]
        
        service_test = TestVectorService()
        service_test.test_vector_service_initialization(vector_config)
        print("   ✓ VectorService initialization test passed (1/1)")
        
        # Run async tests
        print("\n4. Testing VectorService (Async)")
        
        async def run_async_tests():
            service_test = TestVectorService()
            
            # Test add_embeddings 
            print("   Running: test_add_vectors_with_metadata")
            await service_test.test_add_vectors_with_metadata(vector_config, test_vectors, test_metadata)
            print("   ✓ Add vectors test passed")
            
            # Test search
            print("   Running: test_search_vectors_with_metadata")
            await service_test.test_search_vectors_with_metadata(vector_config, test_vectors, test_metadata)
            print("   ✓ Search vectors test passed")
            
            # Test get by ID
            print("   Running: test_get_vector_by_id")
            await service_test.test_get_vector_by_id(vector_config, test_vectors, test_metadata)
            print("   ✓ Get vector by ID test passed")
            
            # Test stats
            print("   Running: test_get_index_stats")
            await service_test.test_get_index_stats(vector_config, test_vectors, test_metadata)
            print("   ✓ Get index stats test passed")
            
            # Test save/load
            print("   Running: test_save_and_load_service")
            await service_test.test_save_and_load_service(vector_config, test_vectors, test_metadata, temp_dir)
            print("   ✓ Save/load service test passed")
            
            # Test clear
            print("   Running: test_clear_index")
            await service_test.test_clear_index(vector_config, test_vectors, test_metadata)
            print("   ✓ Clear index test passed")
            
        asyncio.run(run_async_tests())
        print("   ✓ VectorService async tests passed (6/6)")
        
        # Run integration tests
        print("\n5. Testing VectorService Integration")
        integration_test = TestVectorServiceIntegration()
        
        async def run_integration_tests():
            await integration_test.test_large_dataset_handling()
            print("   ✓ Large dataset handling test passed")
            
            await integration_test.test_different_index_types()
            print("   ✓ Different index types test passed")
            
        asyncio.run(run_integration_tests())
        print("   ✓ VectorService integration tests passed (2/2)")
        
        print("\n" + "=" * 60)
        print("ALL VECTOR SERVICE TESTS PASSED!")
        print("Total: 20/20 tests passed")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_vector_service_tests()
    sys.exit(0 if success else 1)
