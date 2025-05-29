#!/usr/bin/env python3
"""
Simplified Test Suite for LAION Embeddings
Focuses on testing individual components without triggering import conflicts.
"""

import sys
import os
import unittest
import json
import subprocess
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEnvironment(unittest.TestCase):
    """Test the environment setup"""
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def test_required_packages(self):
        """Test that required packages are installed"""
        required_packages = [
            'torch', 'numpy', 'datasets', 'aiohttp', 'requests',
            'transformers', 'faiss', 'pytest'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                self.fail(f"Required package {package} not installed")

class TestFileStructure(unittest.TestCase):
    """Test that all expected files exist"""
    
    def setUp(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_core_files_exist(self):
        """Test that core files exist"""
        core_files = [
            'ipfs_embeddings_py/__init__.py',
            'ipfs_embeddings_py/ipfs_embeddings.py',
            'ipfs_embeddings_py/main_new.py',
            'ipfs_embeddings_py/chunker.py',
            'ipfs_embeddings_py/ipfs_datasets.py',
            'ipfs_embeddings_py/ipfs_multiformats.py',
            'search_embeddings/search_embeddings.py',
            'create_embeddings/create_embeddings.py',
            'shard_embeddings/shard_embeddings.py',
            'sparse_embeddings/sparse_embeddings.py',
            'storacha_clusters/storacha_clusters.py'
        ]
        
        for file_path in core_files:
            full_path = os.path.join(self.base_path, file_path)
            self.assertTrue(os.path.exists(full_path), f"Missing file: {file_path}")
            print(f"✓ {file_path}")
    
    def test_test_files_exist(self):
        """Test that test files exist"""
        test_files = [
            'test/test.py',
            'test/test openvino.py',
            'test/test_openvino2.py',
            'test/test max batch size.py'
        ]
        
        for file_path in test_files:
            full_path = os.path.join(self.base_path, file_path)
            if os.path.exists(full_path):
                print(f"✓ {file_path}")
            else:
                print(f"? {file_path} (not found)")

class TestStandaloneFunctions(unittest.TestCase):
    """Test functions that can be isolated from problematic imports"""
    
    def test_safe_import_main_new(self):
        """Test importing main_new functions directly"""
        try:
            # Try to import individual functions by reading the file
            main_new_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'ipfs_embeddings_py', 'main_new.py'
            )
            
            # Read the file content
            with open(main_new_path, 'r') as f:
                content = f.read()
            
            # Check if key functions are defined
            self.assertIn('def safe_get_cid', content)
            self.assertIn('def index_cid', content)
            self.assertIn('def init_datasets', content)
            print("✓ Main functions found in main_new.py")
            
        except Exception as e:
            self.fail(f"Could not validate main_new.py: {e}")
    
    def test_basic_cid_logic(self):
        """Test basic CID generation logic without imports"""
        import hashlib
        import json
        
        # Simulate the CID generation logic
        test_data = "Hello, world!"
        
        # Basic hash generation (simulating what safe_get_cid might do)
        data_bytes = test_data.encode('utf-8')
        hash_obj = hashlib.sha256(data_bytes)
        hash_hex = hash_obj.hexdigest()
        
        self.assertIsNotNone(hash_hex)
        self.assertEqual(len(hash_hex), 64)  # SHA256 produces 64 char hex
        print(f"✓ Basic hash generation: {hash_hex[:16]}...")
        
        # Test consistency
        hash_obj2 = hashlib.sha256(data_bytes)
        hash_hex2 = hash_obj2.hexdigest()
        self.assertEqual(hash_hex, hash_hex2)
        print("✓ Hash consistency verified")

class TestConfiguration(unittest.TestCase):
    """Test configuration and metadata structures"""
    
    def test_metadata_structure(self):
        """Test that metadata has expected structure"""
        test_metadata = {
            "dataset": "TeraflopAI/Caselaw_Access_Project",
            "column": "text",
            "split": "train",
            "models": [
                "thenlper/gte-small",
                "Alibaba-NLP/gte-large-en-v1.5",
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                "BAAI/bge-m3"
            ],
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
        
        # Validate structure
        self.assertIn("dataset", test_metadata)
        self.assertIn("models", test_metadata)
        self.assertIn("chunk_settings", test_metadata)
        self.assertIsInstance(test_metadata["models"], list)
        self.assertGreater(len(test_metadata["models"]), 0)
        print("✓ Metadata structure validation passed")
    
    def test_resources_structure(self):
        """Test that resources have expected structure"""
        test_resources = {
            "local_endpoints": [
                ["thenlper/gte-small", "cpu", 512],
                ["Alibaba-NLP/gte-large-en-v1.5", "cpu", 8192],
            ],
            "tei_endpoints": [
                ["thenlper/gte-small", "http://127.0.0.1:8080/embed-tiny", 512],
                ["Alibaba-NLP/gte-large-en-v1.5", "http://127.0.0.1:8081/embed-small", 8192],
            ],
            "openvino_endpoints": [
                ["neoALI/bge-m3-rag-ov", "http://127.0.0.1:8090/v2/models/bge-m3-rag-ov/infer", 4095],
            ],
            "libp2p_endpoints": [
                ["thenlper/gte-small", "http://127.0.0.1:8091/embed", 512],
            ]
        }
        
        # Validate structure
        expected_keys = ["local_endpoints", "tei_endpoints", "openvino_endpoints", "libp2p_endpoints"]
        for key in expected_keys:
            self.assertIn(key, test_resources)
            self.assertIsInstance(test_resources[key], list)
        print("✓ Resources structure validation passed")

class TestScriptFiles(unittest.TestCase):
    """Test that script files are executable and valid"""
    
    def setUp(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_bash_scripts_exist(self):
        """Test that bash scripts exist and are executable"""
        bash_scripts = [
            'run.sh', 'search.sh', 'create.sh', 'install_depends.sh',
            'launch_tei.sh', 'autofaiss.sh'
        ]
        
        for script in bash_scripts:
            script_path = os.path.join(self.base_path, script)
            if os.path.exists(script_path):
                # Check if executable
                is_executable = os.access(script_path, os.X_OK)
                print(f"✓ {script} ({'executable' if is_executable else 'not executable'})")
            else:
                print(f"? {script} (not found)")

class TestExistingTests(unittest.TestCase):
    """Test that existing test files can be analyzed"""
    
    def setUp(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_analyze_existing_test_py(self):
        """Analyze the main test.py file"""
        test_path = os.path.join(self.base_path, 'test', 'test.py')
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                content = f.read()
            
            # Check for key test patterns
            test_patterns = [
                'search_embeddings', 'create_embeddings', 'shard_embeddings',
                'sparse_embeddings', 'storacha_clusters'
            ]
            
            found_patterns = []
            for pattern in test_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
            
            print(f"✓ test.py contains tests for: {', '.join(found_patterns)}")
            self.assertGreater(len(found_patterns), 0, "No test patterns found in test.py")
        else:
            self.fail("test.py not found")
    
    def test_analyze_openvino_tests(self):
        """Analyze OpenVINO test files"""
        openvino_tests = ['test openvino.py', 'test_openvino2.py']
        
        for test_file in openvino_tests:
            test_path = os.path.join(self.base_path, 'test', test_file)
            if os.path.exists(test_path):
                with open(test_path, 'r') as f:
                    content = f.read()
                
                # Check for OpenVINO-specific patterns
                if 'openvino' in content.lower() or 'bert' in content.lower():
                    print(f"✓ {test_file} contains OpenVINO/BERT tests")
                else:
                    print(f"? {test_file} exists but no OpenVINO patterns found")
            else:
                print(f"? {test_file} not found")

def run_safe_pytest():
    """Run pytest on existing test files that might work"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = [
        'test/test.py',
        'test/test openvino.py',
        'test/test_openvino2.py',
        'test/test max batch size.py'
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING EXISTING TESTS WITH PYTEST")
    print("=" * 60)
    
    for test_file in test_files:
        test_path = os.path.join(base_path, test_file)
        if os.path.exists(test_path):
            print(f"\nTesting {test_file}:")
            try:
                result = subprocess.run([
                    'python', '-m', 'pytest', test_path, '-v', '--tb=short'
                ], capture_output=True, text=True, cwd=base_path, timeout=60)
                
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                else:
                    print(f"✗ {test_file} failed")
                    if result.stdout:
                        print(f"STDOUT: {result.stdout[-200:]}")  # Last 200 chars
                    if result.stderr:
                        print(f"STDERR: {result.stderr[-200:]}")  # Last 200 chars
                        
            except subprocess.TimeoutExpired:
                print(f"⏱ {test_file} timed out")
            except Exception as e:
                print(f"✗ {test_file} error: {e}")
        else:
            print(f"? {test_file} not found")

def main():
    """Main test runner"""
    print("LAION Embeddings - Simplified Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestEnvironment,
        TestFileStructure,
        TestStandaloneFunctions,
        TestConfiguration,
        TestScriptFiles,
        TestExistingTests
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Try to run existing tests
    run_safe_pytest()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    print("\nRECOMMENDations:")
    if success_rate >= 80:
        print("✓ Core system structure appears valid")
        print("✓ Ready for targeted testing of individual components")
    else:
        print("✗ Fix structural issues before proceeding")
    
    print("\nNEXT STEPS:")
    print("1. Fix import conflicts in main modules")
    print("2. Test individual components in isolation")
    print("3. Create mock-based tests for complex integrations")
    print("4. Set up proper test environment isolation")
    
    return result

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.wasSuccessful() else 1)
