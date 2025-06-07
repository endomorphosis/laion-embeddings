"""
Validation test suite to verify the vector quantization, clustering, and sharding implementation.

This script performs comprehensive validation of all implemented features and provides
a detailed report on the functionality and performance of the vector services.
"""

import asyncio
import sys
import time
from pathlib import Path
import numpy as np
import traceback
from typing import Dict, Any, List
import tempfile
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DIMENSION = 384
TEST_VECTORS_COUNT = 50
PERFORMANCE_VECTORS_COUNT = 200


class ValidationReport:
    """Tracks validation results and generates reports."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_test(self, name: str, status: str, details: Dict[str, Any] = None, error: str = None):
        """Add a test result."""
        self.tests.append({
            'name': name,
            'status': status,
            'details': details or {},
            'error': error,
            'timestamp': time.time()
        })
        
        if status == 'PASS':
            self.passed += 1
        elif status == 'FAIL':
            self.failed += 1
            if error:
                self.errors.append(f"{name}: {error}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        total_tests = self.passed + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        report = [
            "=" * 80,
            "LAION EMBEDDINGS VECTOR SERVICES VALIDATION REPORT",
            "=" * 80,
            f"Total Tests: {total_tests}",
            f"Passed: {self.passed}",
            f"Failed: {self.failed}",
            f"Success Rate: {success_rate:.1f}%",
            "",
            "Test Results:",
            "-" * 40
        ]
        
        for test in self.tests:
            status_symbol = "✓" if test['status'] == 'PASS' else "✗"
            report.append(f"{status_symbol} {test['name']}: {test['status']}")
            
            if test['details']:
                for key, value in test['details'].items():
                    report.append(f"    {key}: {value}")
            
            if test['error']:
                report.append(f"    Error: {test['error']}")
        
        if self.errors:
            report.extend([
                "",
                "Error Summary:",
                "-" * 40
            ])
            for error in self.errors:
                report.append(f"  • {error}")
        
        report.extend([
            "",
            "=" * 80,
            f"Validation {'COMPLETED SUCCESSFULLY' if self.failed == 0 else 'COMPLETED WITH ERRORS'}",
            "=" * 80
        ])
        
        return "\n".join(report)


async def validate_basic_vector_service(report: ValidationReport):
    """Validate basic vector service functionality."""
    try:
        # Test if we can import the service
        try:
            from services.vector_service import VectorService, VectorConfig, create_vector_service
            report.add_test("Import Vector Service", "PASS")
        except ImportError as e:
            report.add_test("Import Vector Service", "FAIL", error=str(e))
            return
        
        # Test service creation
        try:
            config = VectorConfig(dimension=TEST_DIMENSION, index_type="Flat")
            service = VectorService(config)
            report.add_test("Create Vector Service", "PASS")
        except Exception as e:
            report.add_test("Create Vector Service", "FAIL", error=str(e))
            return
        
        # Test adding embeddings
        try:
            test_vectors = np.random.random((10, TEST_DIMENSION)).astype(np.float32)
            test_texts = [f"Test document {i}" for i in range(10)]
            
            result = await service.add_embeddings(
                embeddings=test_vectors,
                texts=test_texts
            )
            
            if result['status'] == 'success' and result['added_count'] == 10:
                report.add_test("Add Embeddings", "PASS", {
                    'added_count': result['added_count'],
                    'total_vectors': result['total_vectors']
                })
            else:
                report.add_test("Add Embeddings", "FAIL", error="Unexpected result format")
        except Exception as e:
            report.add_test("Add Embeddings", "FAIL", error=str(e))
            return
        
        # Test searching
        try:
            query_vector = np.random.random(TEST_DIMENSION).astype(np.float32)
            search_result = await service.search_similar(
                query_embedding=query_vector,
                k=5
            )
            
            if (search_result['status'] == 'success' and 
                len(search_result['results']) <= 5):
                report.add_test("Search Vectors", "PASS", {
                    'results_count': len(search_result['results']),
                    'has_similarity_scores': all('similarity_score' in r for r in search_result['results'])
                })
            else:
                report.add_test("Search Vectors", "FAIL", error="Invalid search results")
        except Exception as e:
            report.add_test("Search Vectors", "FAIL", error=str(e))
        
        # Test factory function
        try:
            factory_service = create_vector_service(dimension=TEST_DIMENSION, index_type="IVF")
            report.add_test("Factory Function", "PASS")
        except Exception as e:
            report.add_test("Factory Function", "FAIL", error=str(e))
            
    except Exception as e:
        report.add_test("Basic Vector Service Validation", "FAIL", error=str(e))


async def validate_ipfs_vector_service(report: ValidationReport):
    """Validate IPFS vector service functionality."""
    try:
        # Test imports
        try:
            from services.ipfs_vector_service import (
                IPFSVectorService, IPFSConfig, DistributedVectorIndex, 
                IPFSVectorStorage, create_ipfs_vector_service
            )
            from services.vector_service import VectorConfig
            report.add_test("Import IPFS Vector Service", "PASS")
        except ImportError as e:
            report.add_test("Import IPFS Vector Service", "FAIL", error=str(e))
            return
        
        # Test configuration creation
        try:
            vector_config = VectorConfig(dimension=TEST_DIMENSION)
            ipfs_config = IPFSConfig(chunk_size=20)
            report.add_test("Create IPFS Configs", "PASS")
        except Exception as e:
            report.add_test("Create IPFS Configs", "FAIL", error=str(e))
            return
        
        # Test service creation (with mocked IPFS)
        try:
            # This will fail if IPFS client is not available, but service should handle gracefully
            service = IPFSVectorService(vector_config, ipfs_config)
            report.add_test("Create IPFS Vector Service", "PASS")
        except ImportError as e:
            # Expected if ipfshttpclient is not installed
            report.add_test("Create IPFS Vector Service", "FAIL", 
                          error="IPFS client not available - this is expected in test environment")
            return
        except Exception as e:
            report.add_test("Create IPFS Vector Service", "FAIL", error=str(e))
            return
        
        # Test distributed index creation
        try:
            dist_index = DistributedVectorIndex(vector_config, ipfs_config)
            report.add_test("Create Distributed Index", "PASS")
        except Exception as e:
            report.add_test("Create Distributed Index", "FAIL", error=str(e))
        
        # Test factory function
        try:
            factory_service = create_ipfs_vector_service(dimension=TEST_DIMENSION)
            report.add_test("IPFS Factory Function", "PASS")
        except Exception as e:
            report.add_test("IPFS Factory Function", "FAIL", error=str(e))
            
    except Exception as e:
        report.add_test("IPFS Vector Service Validation", "FAIL", error=str(e))


async def validate_clustering_service(report: ValidationReport):
    """Validate clustering service functionality."""
    try:
        # Test imports
        try:
            from services.clustering_service import (
                VectorClusterer, SmartShardingService, ClusterConfig, ClusterInfo,
                create_clusterer, create_smart_sharding_service
            )
            from services.vector_service import VectorConfig
            report.add_test("Import Clustering Service", "PASS")
        except ImportError as e:
            report.add_test("Import Clustering Service", "FAIL", error=str(e))
            return
        
        # Test configuration creation
        try:
            cluster_config = ClusterConfig(n_clusters=3, algorithm="kmeans")
            vector_config = VectorConfig(dimension=TEST_DIMENSION)
            report.add_test("Create Clustering Configs", "PASS")
        except Exception as e:
            report.add_test("Create Clustering Configs", "FAIL", error=str(e))
            return
        
        # Test clusterer creation
        try:
            clusterer = VectorClusterer(cluster_config)
            report.add_test("Create Vector Clusterer", "PASS")
        except ImportError as e:
            # Expected if scikit-learn is not installed
            report.add_test("Create Vector Clusterer", "FAIL", 
                          error="scikit-learn not available - this is expected in test environment")
            return
        except Exception as e:
            report.add_test("Create Vector Clusterer", "FAIL", error=str(e))
            return
        
        # Test clustering
        try:
            test_vectors = np.random.random((20, TEST_DIMENSION)).astype(np.float32)
            labels = clusterer.fit_predict(test_vectors)
            
            if len(labels) == 20 and len(np.unique(labels)) <= cluster_config.n_clusters:
                report.add_test("Vector Clustering", "PASS", {
                    'n_vectors': len(labels),
                    'n_clusters': len(np.unique(labels))
                })
            else:
                report.add_test("Vector Clustering", "FAIL", error="Invalid clustering results")
        except Exception as e:
            report.add_test("Vector Clustering", "FAIL", error=str(e))
        
        # Test smart sharding service
        try:
            sharding_service = SmartShardingService(vector_config, cluster_config)
            report.add_test("Create Smart Sharding Service", "PASS")
        except Exception as e:
            report.add_test("Create Smart Sharding Service", "FAIL", error=str(e))
            return
        
        # Test factory functions
        try:
            factory_clusterer = create_clusterer(algorithm="kmeans", n_clusters=3)
            factory_sharding = create_smart_sharding_service(dimension=TEST_DIMENSION, n_clusters=3)
            report.add_test("Clustering Factory Functions", "PASS")
        except Exception as e:
            report.add_test("Clustering Factory Functions", "FAIL", error=str(e))
            
    except Exception as e:
        report.add_test("Clustering Service Validation", "FAIL", error=str(e))


async def validate_integration_workflow(report: ValidationReport):
    """Validate end-to-end integration workflow."""
    try:
        # This test validates that all components can work together
        from services.vector_service import VectorService, VectorConfig
        
        # Test multiple index types
        index_types = ["Flat", "IVF", "HNSW"]
        successful_types = []
        
        for index_type in index_types:
            try:
                config = VectorConfig(dimension=TEST_DIMENSION, index_type=index_type)
                service = VectorService(config)
                
                # Add some test data
                test_vectors = np.random.random((5, TEST_DIMENSION)).astype(np.float32)
                result = await service.add_embeddings(embeddings=test_vectors)
                
                if result['status'] == 'success':
                    successful_types.append(index_type)
            except Exception:
                continue
        
        if successful_types:
            report.add_test("Multiple Index Types", "PASS", {
                'successful_types': successful_types,
                'total_tested': len(index_types)
            })
        else:
            report.add_test("Multiple Index Types", "FAIL", error="No index types worked")
        
        # Test service integration
        try:
            # Create services with compatible configurations
            vector_config = VectorConfig(dimension=TEST_DIMENSION, index_type="Flat")
            
            # Basic service
            basic_service = VectorService(vector_config)
            
            # Add data to basic service
            test_vectors = np.random.random((15, TEST_DIMENSION)).astype(np.float32)
            test_texts = [f"Integration test document {i}" for i in range(15)]
            
            basic_result = await basic_service.add_embeddings(
                embeddings=test_vectors,
                texts=test_texts
            )
            
            # Search in basic service
            query_vector = np.random.random(TEST_DIMENSION).astype(np.float32)
            search_result = await basic_service.search_similar(
                query_embedding=query_vector,
                k=5
            )
            
            if (basic_result['status'] == 'success' and 
                search_result['status'] == 'success'):
                report.add_test("Service Integration", "PASS", {
                    'vectors_added': basic_result['added_count'],
                    'search_results': len(search_result['results'])
                })
            else:
                report.add_test("Service Integration", "FAIL", error="Integration workflow failed")
        except Exception as e:
            report.add_test("Service Integration", "FAIL", error=str(e))
            
    except Exception as e:
        report.add_test("Integration Workflow Validation", "FAIL", error=str(e))


async def validate_performance(report: ValidationReport):
    """Validate performance characteristics."""
    try:
        from services.vector_service import VectorService, VectorConfig
        
        # Test with larger dataset
        config = VectorConfig(dimension=TEST_DIMENSION, index_type="Flat", batch_size=50)
        service = VectorService(config)
        
        # Generate performance test data
        large_vectors = np.random.random((PERFORMANCE_VECTORS_COUNT, TEST_DIMENSION)).astype(np.float32)
        
        # Measure add performance
        start_time = time.time()
        result = await service.add_embeddings(embeddings=large_vectors)
        add_time = time.time() - start_time
        
        if result['status'] == 'success':
            # Measure search performance
            query_vector = np.random.random(TEST_DIMENSION).astype(np.float32)
            
            search_times = []
            for _ in range(5):  # Multiple searches for average
                start_time = time.time()
                search_result = await service.search_similar(query_embedding=query_vector, k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            
            report.add_test("Performance Test", "PASS", {
                'vectors_count': PERFORMANCE_VECTORS_COUNT,
                'add_time_sec': round(add_time, 3),
                'avg_search_time_sec': round(avg_search_time, 4),
                'vectors_per_sec': round(PERFORMANCE_VECTORS_COUNT / add_time, 2)
            })
        else:
            report.add_test("Performance Test", "FAIL", error="Failed to add vectors for performance test")
            
    except Exception as e:
        report.add_test("Performance Validation", "FAIL", error=str(e))


async def validate_error_handling(report: ValidationReport):
    """Validate error handling and edge cases."""
    try:
        from services.vector_service import VectorService, VectorConfig
        
        config = VectorConfig(dimension=TEST_DIMENSION)
        service = VectorService(config)
        
        # Test dimension mismatch
        try:
            wrong_dimension_vectors = np.random.random((5, TEST_DIMENSION + 10)).astype(np.float32)
            result = await service.add_embeddings(embeddings=wrong_dimension_vectors)
            
            # This should raise an error
            report.add_test("Dimension Mismatch Handling", "FAIL", 
                          error="Should have raised dimension mismatch error")
        except ValueError:
            # Expected behavior
            report.add_test("Dimension Mismatch Handling", "PASS")
        except Exception as e:
            report.add_test("Dimension Mismatch Handling", "FAIL", error=str(e))
        
        # Test empty search
        try:
            query_vector = np.random.random(TEST_DIMENSION).astype(np.float32)
            search_result = await service.search_similar(query_embedding=query_vector, k=5)
            
            # Should handle gracefully even with no data
            if search_result['status'] == 'success':
                report.add_test("Empty Index Search", "PASS")
            else:
                report.add_test("Empty Index Search", "FAIL", error="Failed to handle empty index search")
        except Exception as e:
            report.add_test("Empty Index Search", "FAIL", error=str(e))
        
        # Test invalid index type
        try:
            invalid_config = VectorConfig(dimension=TEST_DIMENSION, index_type="INVALID_TYPE")
            invalid_service = VectorService(invalid_config)
            report.add_test("Invalid Index Type Handling", "FAIL", 
                          error="Should have raised error for invalid index type")
        except ValueError:
            report.add_test("Invalid Index Type Handling", "PASS")
        except Exception as e:
            report.add_test("Invalid Index Type Handling", "FAIL", error=str(e))
            
    except Exception as e:
        report.add_test("Error Handling Validation", "FAIL", error=str(e))


async def main():
    """Run comprehensive validation of the vector services implementation."""
    print("Starting LAION Embeddings Vector Services Validation...")
    print("=" * 80)
    
    report = ValidationReport()
    
    # Run all validation tests
    validation_functions = [
        validate_basic_vector_service,
        validate_ipfs_vector_service,
        validate_clustering_service,
        validate_integration_workflow,
        validate_performance,
        validate_error_handling
    ]
    
    for validate_func in validation_functions:
        try:
            print(f"Running {validate_func.__name__}...")
            await validate_func(report)
        except Exception as e:
            print(f"Error in {validate_func.__name__}: {e}")
            traceback.print_exc()
    
    # Generate and display report
    print("\n" + report.generate_report())
    
    # Return exit code based on results
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
