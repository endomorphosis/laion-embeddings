# Dependency Resolution Strategy
## PyTorch/Torchvision Import Conflict Resolution

### Problem Analysis
The current issues stem from:
1. **Version conflicts** between torch (2.7.0), torchvision (0.19.0), and transformers (4.46.0)
2. **Circular imports** in the transformers → torchvision chain
3. **Missing operators** in torchvision._meta_registrations

### Resolution Approach

#### Option 1: Version Alignment (Recommended)
Update to compatible versions that work together:

```bash
# Uninstall problematic packages
pip uninstall torch torchvision torchaudio transformers

# Install compatible versions
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install transformers==4.44.0

# Alternative: Use PyTorch index for guaranteed compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option 2: Import Isolation (Backup)
If version updates don't work, implement import isolation:

```python
# Create import_manager.py
import sys
import importlib
from typing import Any, Dict
from unittest.mock import Mock

class SafeImportManager:
    """Manages safe imports with fallbacks for problematic modules."""
    
    def __init__(self):
        self.mock_modules = {}
        self.failed_imports = set()
    
    def safe_import(self, module_name: str, fallback_mock: bool = True) -> Any:
        """Safely import a module with optional mock fallback."""
        try:
            return importlib.import_module(module_name)
        except (ImportError, RuntimeError) as e:
            self.failed_imports.add(module_name)
            if fallback_mock:
                mock_module = Mock()
                mock_module.__name__ = module_name
                sys.modules[module_name] = mock_module
                return mock_module
            raise e
    
    def preload_mocks(self):
        """Preload mocks for known problematic modules."""
        problematic_modules = [
            'torchvision._meta_registrations',
            'torchvision.extension',
            'ipfs_accelerate_py',
            'ipfs_accelerate_py.worker.skillset.hf_whisper',
            'ipfs_accelerate_py.worker.skillset.hf_xclip',
        ]
        
        for module_name in problematic_modules:
            if module_name not in sys.modules:
                sys.modules[module_name] = Mock()

# Usage in main.py or test files
import_manager = SafeImportManager()
import_manager.preload_mocks()

# Safe imports
transformers = import_manager.safe_import('transformers')
torch = import_manager.safe_import('torch')
```

#### Option 3: Lazy Import Pattern
Implement lazy imports to defer problematic imports:

```python
# lazy_imports.py
from typing import Optional, Any
import importlib

class LazyImport:
    """Lazy import that only loads when accessed."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Optional[Any] = None
    
    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return self._module(*args, **kwargs)

# Usage in modules
transformers = LazyImport('transformers')
torch = LazyImport('torch')
torchvision = LazyImport('torchvision')

# These won't import until actually used
def some_function():
    model = transformers.AutoModel.from_pretrained('model_name')
    return model
```

### Implementation Steps

#### Step 1: Clean Environment
```bash
# Create backup of current requirements
cp requirements.txt requirements.backup.txt

# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate

# Install core dependencies first
pip install fastapi uvicorn pydantic
pip install numpy pandas
```

#### Step 2: Install PyTorch Stack
```bash
# Install compatible PyTorch versions
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# Verify installation
python -c "import torch; import torchvision; print('PyTorch OK')"
```

#### Step 3: Install Additional Dependencies
```bash
# Install transformers with compatible version
pip install transformers==4.44.0

# Install other ML libraries
pip install sentence-transformers faiss-cpu
pip install datasets huggingface-hub

# Test compatibility
python -c "
import torch
import torchvision
import transformers
print('All imports successful')
"
```

#### Step 4: Update Requirements File
```bash
# Generate new requirements with compatible versions
pip freeze > requirements_fixed.txt

# Test installation from scratch
deactivate
rm -rf test_env
python -m venv test_env
source test_env/bin/activate
pip install -r requirements_fixed.txt
```

### Testing Strategy

#### Create Test Import Script
```python
# test_imports.py
import sys
import traceback

def test_imports():
    """Test all critical imports and report issues."""
    import_tests = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('Basic torch ops', 'torch.tensor([1.0])'),
        ('Torchvision transforms', 'torchvision.transforms.ToTensor()'),
        ('Transformers AutoModel', 'transformers.AutoModel'),
        ('Main application', 'main'),
    ]
    
    results = {}
    for name, import_statement in import_tests:
        try:
            if name == 'Main application':
                import main
                results[name] = "SUCCESS"
            else:
                exec(f"import {import_statement}")
                results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            traceback.print_exc()
    
    return results

if __name__ == "__main__":
    results = test_imports()
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
```

### Rollback Plan
If dependency updates cause other issues:

1. **Immediate Rollback**:
   ```bash
   pip install -r requirements.backup.txt
   ```

2. **Alternative Approach**:
   - Use Docker containers with fixed environments
   - Implement comprehensive mocking for tests
   - Create separate environments for different components

### Success Criteria
- [ ] All imports in test_imports.py pass
- [ ] pytest runs without import errors
- [ ] Main application starts successfully
- [ ] Basic API endpoints respond correctly
- [ ] No regression in existing functionality

This strategy provides multiple approaches to resolve the dependency issues while maintaining system functionality.
