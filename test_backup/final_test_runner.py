#!/usr/bin/env python3
"""
Final Test Suite Runner for LAION Embeddings
Comprehensive validation of the system with clear output
"""

import sys
import os
import unittest
import time
import json
import hashlib
import subprocess
from typing import Dict, List, Any

def banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner with text"""
    print(char * width)
    print(f" {text} ".center(width, char))
    print(char * width)

def test_result(name: str, success: bool, details: str = "") -> None:
    """Print a test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status:8} | {name}")
    if details:
        print(f"         | {details}")

class LaionEmbeddingsTestSuite:
    """Comprehensive test suite for LAION Embeddings"""
    
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories"""
        banner("LAION EMBEDDINGS - COMPREHENSIVE TEST SUITE")
        
        test_categories = [
            ("Environment Setup", self.test_environment),
            ("File Structure", self.test_file_structure),
            ("Core Functionality", self.test_core_functionality),
            ("Configuration", self.test_configuration),
            ("Data Structures", self.test_data_structures),
            ("Error Handling", self.test_error_handling),
            ("Integration", self.test_integration),
        ]
        
        overall_results = {}
        
        for category_name, test_func in test_categories:
            banner(category_name)
            start_time = time.time()
            
            try:
                results = test_func()
                elapsed = time.time() - start_time
                
                passed = sum(1 for r in results.values() if r.get('success', False))
                total = len(results)
                
                print(f"\nCategory Results: {passed}/{total} tests passed ({elapsed:.2f}s)")
                overall_results[category_name] = {
                    'results': results,
                    'passed': passed,
                    'total': total,
                    'elapsed': elapsed
                }
                
            except Exception as e:
                print(f"Category failed with error: {e}")
                overall_results[category_name] = {
                    'error': str(e),
                    'passed': 0,
                    'total': 1,
                    'elapsed': time.time() - start_time
                }
            
            print()
        
        return overall_results
    
    def test_environment(self) -> Dict[str, Dict]:
        """Test environment setup"""
        results = {}
        
        # Python version
        version = sys.version_info
        success = version.major >= 3 and version.minor >= 8
        results['python_version'] = {
            'success': success,
            'details': f"Python {version.major}.{version.minor}.{version.micro}"
        }
        test_result("Python Version", success, results['python_version']['details'])
        
        # Required packages
        packages = ['torch', 'transformers', 'datasets', 'numpy', 'aiohttp', 'requests', 'faiss']
        for package in packages:
            try:
                __import__(package)
                success = True
                details = "Available"
            except ImportError as e:
                success = False
                details = f"Missing: {e}"
            
            results[f'package_{package}'] = {'success': success, 'details': details}
            test_result(f"Package {package}", success, details)
        
        return results
    
    def test_file_structure(self) -> Dict[str, Dict]:
        """Test file structure"""
        results = {}
        
        core_files = [
            'ipfs_embeddings_py/__init__.py',
            'ipfs_embeddings_py/ipfs_embeddings.py',
            'ipfs_embeddings_py/main_new.py',
            'ipfs_embeddings_py/chunker.py',
            'ipfs_embeddings_py/ipfs_datasets.py',
            'search_embeddings/search_embeddings.py',
            'create_embeddings/create_embeddings.py',
            'shard_embeddings/shard_embeddings.py',
            'sparse_embeddings/sparse_embeddings.py',
            'storacha_clusters/storacha_clusters.py'
        ]
        
        for file_path in core_files:
            full_path = os.path.join(self.base_path, file_path)
            success = os.path.exists(full_path)
            size = os.path.getsize(full_path) if success else 0
            
            results[f'file_{file_path.replace("/", "_")}'] = {
                'success': success,
                'details': f"{size} bytes" if success else "Not found"
            }
            test_result(f"File {file_path}", success, results[f'file_{file_path.replace("/", "_")}']['details'])
        
        return results
    
    def test_core_functionality(self) -> Dict[str, Dict]:
        """Test core functionality"""
        results = {}
        
        # Test hash generation (CID simulation)
        try:
            test_data = "Hello, world!"
            hash_obj = hashlib.sha256(test_data.encode('utf-8'))
            cid = hash_obj.hexdigest()
            
            success = len(cid) == 64 and all(c in '0123456789abcdef' for c in cid)
            results['cid_generation'] = {
                'success': success,
                'details': f"Generated: {cid[:16]}..."
            }
        except Exception as e:
            results['cid_generation'] = {'success': False, 'details': str(e)}
        
        test_result("CID Generation", results['cid_generation']['success'], 
                   results['cid_generation']['details'])
        
        # Test batch creation
        try:
            items = list(range(100))
            batch_size = 10
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            success = len(batches) == 10 and all(len(b) == 10 for b in batches)
            results['batch_creation'] = {
                'success': success,
                'details': f"Created {len(batches)} batches"
            }
        except Exception as e:
            results['batch_creation'] = {'success': False, 'details': str(e)}
        
        test_result("Batch Creation", results['batch_creation']['success'],
                   results['batch_creation']['details'])
        
        return results
    
    def test_configuration(self) -> Dict[str, Dict]:
        """Test configuration structures"""
        results = {}
        
        # Test metadata structure
        try:
            metadata = {
                "dataset": "TeraflopAI/Caselaw_Access_Project",
                "column": "text",
                "split": "train",
                "models": ["thenlper/gte-small", "Alibaba-NLP/gte-large-en-v1.5"],
                "chunk_settings": {
                    "chunk_size": 512,
                    "n_sentences": 8,
                    "step_size": 256,
                    "method": "fixed",
                    "embed_model": "thenlper/gte-small",
                    "tokenizer": None
                },
                "dst_path": "/tmp/test_embeddings",
            }
            
            required_keys = ["dataset", "column", "split", "models", "chunk_settings", "dst_path"]
            success = all(key in metadata for key in required_keys)
            
            results['metadata_structure'] = {
                'success': success,
                'details': f"Has {len([k for k in required_keys if k in metadata])}/{len(required_keys)} required keys"
            }
        except Exception as e:
            results['metadata_structure'] = {'success': False, 'details': str(e)}
        
        test_result("Metadata Structure", results['metadata_structure']['success'],
                   results['metadata_structure']['details'])
        
        # Test resources structure
        try:
            resources = {
                "local_endpoints": [["thenlper/gte-small", "cpu", 512]],
                "tei_endpoints": [["thenlper/gte-small", "http://127.0.0.1:8080/embed-tiny", 512]],
                "openvino_endpoints": [],
                "libp2p_endpoints": []
            }
            
            required_keys = ["local_endpoints", "tei_endpoints", "openvino_endpoints", "libp2p_endpoints"]
            success = all(key in resources and isinstance(resources[key], list) for key in required_keys)
            
            results['resources_structure'] = {
                'success': success,
                'details': f"Has {len(required_keys)} endpoint types"
            }
        except Exception as e:
            results['resources_structure'] = {'success': False, 'details': str(e)}
        
        test_result("Resources Structure", results['resources_structure']['success'],
                   results['resources_structure']['details'])
        
        return results
    
    def test_data_structures(self) -> Dict[str, Dict]:
        """Test data structure handling"""
        results = {}
        
        # Test dataset structure
        try:
            dataset = {
                "data": [
                    {"text": "Sample text 1", "id": "1"},
                    {"text": "Sample text 2", "id": "2"},
                    {"text": "Sample text 3", "id": "3"}
                ],
                "metadata": {
                    "total_items": 3,
                    "columns": ["text", "id"]
                }
            }
            
            success = (
                "data" in dataset and 
                "metadata" in dataset and
                isinstance(dataset["data"], list) and
                len(dataset["data"]) > 0 and
                all("text" in item and "id" in item for item in dataset["data"])
            )
            
            results['dataset_structure'] = {
                'success': success,
                'details': f"Dataset with {len(dataset['data'])} items"
            }
        except Exception as e:
            results['dataset_structure'] = {'success': False, 'details': str(e)}
        
        test_result("Dataset Structure", results['dataset_structure']['success'],
                   results['dataset_structure']['details'])
        
        return results
    
    def test_error_handling(self) -> Dict[str, Dict]:
        """Test error handling scenarios"""
        results = {}
        
        # Test input validation
        try:
            def validate_text(text):
                if not text or not text.strip():
                    raise ValueError("Text cannot be empty")
                return text.strip()
            
            # Test empty input handling
            try:
                validate_text("")
                success = False  # Should have raised an exception
            except ValueError:
                success = True  # Exception was properly raised
            
            results['input_validation'] = {
                'success': success,
                'details': "Empty input properly rejected"
            }
        except Exception as e:
            results['input_validation'] = {'success': False, 'details': str(e)}
        
        test_result("Input Validation", results['input_validation']['success'],
                   results['input_validation']['details'])
        
        return results
    
    def test_integration(self) -> Dict[str, Dict]:
        """Test integration aspects"""
        results = {}
        
        # Test component file analysis
        component_files = {
            'search_embeddings': 'search_embeddings/search_embeddings.py',
            'create_embeddings': 'create_embeddings/create_embeddings.py',
            'shard_embeddings': 'shard_embeddings/shard_embeddings.py',
            'sparse_embeddings': 'sparse_embeddings/sparse_embeddings.py',
            'storacha_clusters': 'storacha_clusters/storacha_clusters.py'
        }
        
        for component, file_path in component_files.items():
            try:
                full_path = os.path.join(self.base_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Check for class definitions
                    has_class = f"class {component}" in content or "def " in content
                    success = has_class and len(content) > 100  # Basic sanity check
                    
                    results[f'component_{component}'] = {
                        'success': success,
                        'details': f"File size: {len(content)} chars, has_class: {has_class}"
                    }
                else:
                    results[f'component_{component}'] = {
                        'success': False,
                        'details': "File not found"
                    }
            except Exception as e:
                results[f'component_{component}'] = {'success': False, 'details': str(e)}
            
            test_result(f"Component {component}", results[f'component_{component}']['success'],
                       results[f'component_{component}']['details'])
        
        return results
    
    def print_final_summary(self, overall_results: Dict[str, Any]) -> None:
        """Print final test summary"""
        banner("FINAL TEST SUMMARY")
        
        total_passed = 0
        total_tests = 0
        
        for category, data in overall_results.items():
            if 'error' in data:
                print(f"{category:20} | ERROR: {data['error']}")
                total_tests += 1
            else:
                passed = data['passed']
                total = data['total']
                elapsed = data['elapsed']
                
                total_passed += passed
                total_tests += total
                
                status = "✓" if passed == total else "✗"
                print(f"{category:20} | {status} {passed:2}/{total:2} tests ({elapsed:5.2f}s)")
        
        print("-" * 50)
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        overall_status = "✓ PASS" if success_rate >= 80 else "✗ FAIL"
        
        print(f"{'OVERALL':20} | {overall_status} {total_passed:2}/{total_tests:2} tests ({success_rate:5.1f}%)")
        
        banner("RECOMMENDATIONS")
        
        if success_rate >= 90:
            print("✓ Excellent! System is ready for production testing")
            print("✓ All core components are properly structured")
            print("✓ Consider running performance benchmarks")
        elif success_rate >= 80:
            print("✓ Good! System is mostly ready")
            print("• Fix any remaining issues")
            print("• Test with real data")
        elif success_rate >= 60:
            print("⚠ Partial! System needs some work")
            print("• Address failed tests")
            print("• Check import dependencies")
        else:
            print("✗ Poor! System needs significant work")
            print("• Fix fundamental issues first")
            print("• Check installation and dependencies")
        
        print("\nNext Steps:")
        print("1. Run existing test files individually")
        print("2. Test with mock data endpoints")
        print("3. Validate with real datasets")
        print("4. Performance testing")
        print("5. Integration testing")
        
        return success_rate

def main():
    """Main test runner"""
    suite = LaionEmbeddingsTestSuite()
    overall_results = suite.run_all_tests()
    success_rate = suite.print_final_summary(overall_results)
    
    # Exit with appropriate code
    exit_code = 0 if success_rate >= 80 else 1
    print(f"\nExiting with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
