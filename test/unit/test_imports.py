#!/usr/bin/env python3
"""
Test script to diagnose and validate import issues in the LAION Embeddings project.
This script systematically tests imports to identify specific conflict points.
"""

import sys
import traceback
import importlib
from typing import Dict, List, Tuple
import subprocess
import pkg_resources

def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    key_packages = [
        'torch', 'torchvision', 'torchaudio', 'transformers', 
        'fastapi', 'uvicorn', 'pydantic', 'numpy', 'pandas'
    ]
    
    versions = {}
    for package in key_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
        except pkg_resources.DistributionNotFound:
            versions[package] = "NOT INSTALLED"
        except Exception as e:
            versions[package] = f"ERROR: {e}"
    
    return versions

def test_basic_imports() -> Dict[str, str]:
    """Test basic Python and system imports."""
    basic_tests = [
        ('Python sys', 'sys'),
        ('Python os', 'os'),
        ('Python json', 'json'),
        ('Python typing', 'typing'),
        ('Numpy', 'numpy'),
        ('Pandas', 'pandas'),
    ]
    
    results = {}
    for name, module in basic_tests:
        try:
            importlib.import_module(module)
            results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
    
    return results

def test_pytorch_stack() -> Dict[str, str]:
    """Test PyTorch ecosystem imports."""
    pytorch_tests = [
        ('torch', 'torch'),
        ('torch tensor creation', 'torch.tensor([1.0])'),
        ('torchaudio', 'torchaudio'),
    ]
    
    results = {}
    for name, test in pytorch_tests:
        try:
            if name == 'torch tensor creation':
                import torch
                tensor = torch.tensor([1.0])
                results[name] = "SUCCESS"
            else:
                importlib.import_module(test)
                results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
    
    return results

def test_torchvision_components() -> Dict[str, str]:
    """Test torchvision imports step by step."""
    torchvision_tests = [
        ('torchvision base', 'torchvision'),
        ('torchvision.transforms', 'torchvision.transforms'),
        ('torchvision.models', 'torchvision.models'),
        ('torchvision._meta_registrations', 'torchvision._meta_registrations'),
        ('torchvision.extension', 'torchvision.extension'),
    ]
    
    results = {}
    for name, module in torchvision_tests:
        try:
            importlib.import_module(module)
            results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
    
    return results

def test_transformers_components() -> Dict[str, str]:
    """Test transformers library imports."""
    transformers_tests = [
        ('transformers base', 'transformers'),
        ('transformers.AutoModel', 'transformers.AutoModel'),
        ('transformers.AutoTokenizer', 'transformers.AutoTokenizer'),
        ('transformers.models.whisper', 'transformers.models.whisper'),
        ('transformers.processing_utils', 'transformers.processing_utils'),
    ]
    
    results = {}
    for name, component in transformers_tests:
        try:
            if '.' in component and component != 'transformers.models.whisper':
                module_path, attr_name = component.rsplit('.', 1)
                module = importlib.import_module(module_path)
                getattr(module, attr_name)
                results[name] = "SUCCESS"
            else:
                importlib.import_module(component)
                results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
    
    return results

def test_project_imports() -> Dict[str, str]:
    """Test project-specific imports."""
    project_tests = [
        ('FastAPI imports', 'fastapi'),
        ('Pydantic imports', 'pydantic'),
        ('Search embeddings', 'search_embeddings'),
        ('Create embeddings', 'create_embeddings'),
        ('IPFS embeddings py', 'ipfs_embeddings_py'),
        ('Main application', 'main'),
    ]
    
    results = {}
    for name, module in project_tests:
        try:
            importlib.import_module(module)
            results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            if name == 'Main application':
                # Capture full traceback for main import failure
                results[f'{name} (traceback)'] = traceback.format_exc()
    
    return results

def suggest_fixes(all_results: Dict[str, Dict[str, str]]) -> List[str]:
    """Suggest fixes based on test results."""
    suggestions = []
    
    # Check for PyTorch issues
    pytorch_results = all_results.get('PyTorch Stack', {})
    if any('FAILED' in result for result in pytorch_results.values()):
        suggestions.append(
            "🔧 PyTorch Issue Detected:\n"
            "   Try: pip uninstall torch torchvision torchaudio && "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        )
    
    # Check for torchvision issues
    torchvision_results = all_results.get('Torchvision Components', {})
    if any('operator torchvision::nms does not exist' in result for result in torchvision_results.values()):
        suggestions.append(
            "🔧 Torchvision NMS Issue Detected:\n"
            "   Try: pip install --upgrade --force-reinstall torchvision"
        )
    
    # Check for transformers issues
    transformers_results = all_results.get('Transformers Components', {})
    if any('partially initialized module' in result for result in transformers_results.values()):
        suggestions.append(
            "🔧 Circular Import Issue Detected:\n"
            "   Try: Implement lazy imports or mock problematic modules in tests"
        )
    
    # Check for project import issues
    project_results = all_results.get('Project Imports', {})
    if any('FAILED' in result for result in project_results.values()):
        suggestions.append(
            "🔧 Project Import Issue Detected:\n"
            "   Try: Update PYTHONPATH or implement import isolation"
        )
    
    return suggestions

def main():
    """Run all diagnostic tests and provide recommendations."""
    print("🔍 LAION Embeddings - Import Diagnostic Tool")
    print("=" * 60)
    
    # Get package versions
    print("\n📦 Package Versions:")
    versions = get_package_versions()
    for package, version in versions.items():
        status = "✅" if version != "NOT INSTALLED" and "ERROR" not in version else "❌"
        print(f"   {status} {package}: {version}")
    
    # Run all tests
    all_tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Stack", test_pytorch_stack),
        ("Torchvision Components", test_torchvision_components),
        ("Transformers Components", test_transformers_components),
        ("Project Imports", test_project_imports),
    ]
    
    all_results = {}
    for test_name, test_func in all_tests:
        print(f"\n🧪 Testing {test_name}:")
        results = test_func()
        all_results[test_name] = results
        
        for item_name, result in results.items():
            status = "✅" if result == "SUCCESS" else "❌"
            print(f"   {status} {item_name}: {result}")
    
    # Provide suggestions
    print(f"\n💡 Recommendations:")
    suggestions = suggest_fixes(all_results)
    if suggestions:
        for suggestion in suggestions:
            print(f"   {suggestion}")
    else:
        print("   🎉 All tests passed! No issues detected.")
    
    # Summary
    total_tests = sum(len(results) for results in all_results.values())
    failed_tests = sum(
        len([r for r in results.values() if 'FAILED' in r]) 
        for results in all_results.values()
    )
    
    print(f"\n📊 Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {total_tests - failed_tests}")
    print(f"   Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("   🎉 All imports working correctly!")
        return 0
    else:
        print("   ⚠️  Issues detected - see recommendations above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
