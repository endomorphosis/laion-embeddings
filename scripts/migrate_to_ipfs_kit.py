#!/usr/bin/env python3
"""
Migration Script: Replace old IPFS packages with new ipfs_kit_py
This script migrates from old ipfs_*_py packages to the consolidated ipfs_kit_py
"""

import os
import re
import shutil
import json
from pathlib import Path
from typing import Dict, List

class IPFSKitMigrator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_deprecated_ipfs"
        self.ipfs_kit_source = self.project_root / "docs" / "ipfs_kit_py"
        
        # Packages to deprecate and replace
        self.deprecated_packages = [
            'ipfs_datasets_py',
            'ipfs_embeddings_py', 
            'ipfs_accelerate_py',
            'ipfs_transformers_py',
            'ipfshttpclient'
        ]
        
        # Import mappings from old to new
        self.import_mappings = {
            r'from\s+ipfs_embeddings_py\s+import\s+ipfs_embeddings_py': 'from ipfs_kit_py import ipfs_kit',
            r'from\s+ipfs_datasets_py\s+import': 'from ipfs_kit_py import ipfs_kit',
            r'from\s+ipfs_accelerate_py\s+import': 'from ipfs_kit_py import ipfs_kit',
            r'from\s+ipfs_transformers_py\s+import': 'from ipfs_kit_py import ipfs_kit',
            r'import\s+ipfs_embeddings_py': 'from ipfs_kit_py import ipfs_kit',
            r'import\s+ipfs_datasets_py': 'from ipfs_kit_py import ipfs_kit',
            r'import\s+ipfshttpclient': 'from ipfs_kit_py import ipfs_kit',
        }
        
        # Function call mappings
        self.function_mappings = {
            r'ipfs_embeddings_py\.': 'ipfs_kit.',
            r'ipfs_datasets_py\.': 'ipfs_kit.',
            r'ipfs_accelerate_py\.': 'ipfs_kit.',
            r'ipfs_transformers_py\.': 'ipfs_kit.',
        }

    def create_backup(self):
        """Create backup of current state before migration."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            shutil.copy2(req_file, self.backup_dir / "requirements.txt.backup")
        
        print(f"Created backup directory: {self.backup_dir}")

    def update_requirements(self):
        """Update requirements.txt to use new ipfs_kit_py version."""
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            print("requirements.txt not found!")
            return False
        
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        removed_packages = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('#'):
                new_lines.append(line)
                continue
            
            # Check if this is a deprecated package
            package_name = line_stripped.split('==')[0].split('>=')[0].split('<=')[0].strip()
            
            if package_name in self.deprecated_packages:
                removed_packages.append(package_name)
                new_lines.append(f"# DEPRECATED: {line_stripped} - replaced with ipfs_kit_py\n")
            elif package_name == 'ipfs_kit_py':
                # Update to use local development version
                new_lines.append("# ipfs_kit_py - using local development version from docs/ipfs_kit_py\n")
            else:
                new_lines.append(line)
        
        # Write updated requirements
        with open(req_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Updated requirements.txt:")
        print(f"  Deprecated packages: {removed_packages}")
        return True

    def migrate_imports_in_file(self, file_path: Path) -> bool:
        """Migrate imports in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False
        
        original_content = content
        modified = False
        
        # Apply import mappings
        for old_pattern, new_import in self.import_mappings.items():
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_import, content)
                modified = True
        
        # Apply function call mappings
        for old_pattern, new_call in self.function_mappings.items():
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_call, content)
                modified = True
        
        # Write back if modified
        if modified:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated: {file_path}")
                return True
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False
        
        return False

    def migrate_python_files(self):
        """Migrate all Python files in the project."""
        python_files = list(self.project_root.glob("**/*.py"))
        
        # Exclude certain directories
        exclude_patterns = [
            "backup_deprecated_ipfs",
            "docs/ipfs_kit_py",
            "__pycache__",
            ".git",
            ".pytest_cache"
        ]
        
        migrated_files = []
        
        for py_file in python_files:
            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            
            if self.migrate_imports_in_file(py_file):
                migrated_files.append(py_file)
        
        print(f"\nMigrated {len(migrated_files)} Python files")
        return migrated_files

    def install_local_ipfs_kit(self):
        """Install the local ipfs_kit_py package in development mode."""
        if not self.ipfs_kit_source.exists():
            print(f"Error: ipfs_kit_py source not found at {self.ipfs_kit_source}")
            return False
        
        try:
            # Check if we're in a virtual environment
            import subprocess
            import sys
            
            # Use the current Python executable (which will be in venv if activated)
            python_exe = sys.executable
            
            result = subprocess.run([
                python_exe, '-m', 'pip', 'install', '-e', str(self.ipfs_kit_source)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("Successfully installed local ipfs_kit_py in development mode")
                print(f"Using Python: {python_exe}")
                return True
            else:
                print(f"Error installing ipfs_kit_py: {result.stderr}")
                # Try with --break-system-packages if in system Python
                if "externally-managed-environment" in result.stderr:
                    print("Trying with --break-system-packages...")
                    result2 = subprocess.run([
                        python_exe, '-m', 'pip', 'install', '-e', str(self.ipfs_kit_source), '--break-system-packages'
                    ], capture_output=True, text=True, cwd=self.project_root)
                    
                    if result2.returncode == 0:
                        print("Successfully installed with --break-system-packages")
                        return True
                    else:
                        print(f"Still failed: {result2.stderr}")
                return False
                
        except Exception as e:
            print(f"Error installing ipfs_kit_py: {e}")
            return False

    def update_conftest_mocks(self):
        """Update conftest.py to mock the new ipfs_kit_py imports."""
        conftest_file = self.project_root / "conftest.py"
        
        if not conftest_file.exists():
            return
        
        try:
            with open(conftest_file, 'r') as f:
                content = f.read()
            
            # Add updated mocks for the new ipfs_kit_py structure
            mock_additions = '''
# Updated mocks for consolidated ipfs_kit_py
sys.modules['ipfs_kit_py.ipfs_kit'] = MagicMock()
sys.modules['ipfs_kit_py.storacha_kit'] = MagicMock()  
sys.modules['ipfs_kit_py.s3_kit'] = MagicMock()
sys.modules['ipfs_kit_py.high_level_api'] = MagicMock()
'''
            
            # Check if we need to add the new mocks
            if 'ipfs_kit_py.storacha_kit' not in content:
                # Find the existing mock section and add our mocks
                mock_section = re.search(r'(sys\.modules\[.*ipfs_kit_py.*\].*)', content)
                if mock_section:
                    # Add after existing mocks
                    insertion_point = mock_section.end()
                    content = content[:insertion_point] + mock_additions + content[insertion_point:]
                    
                    with open(conftest_file, 'w') as f:
                        f.write(content)
                    
                    print("Updated conftest.py with new mocks")
        
        except Exception as e:
            print(f"Error updating conftest.py: {e}")

    def create_migration_report(self, migrated_files: List[Path]):
        """Create a report of the migration."""
        report = f"""# IPFS Kit Migration Report

## Migration Summary
- Date: {__import__('datetime').datetime.now().isoformat()}
- Project: {self.project_root}
- Deprecated packages: {', '.join(self.deprecated_packages)}
- Files migrated: {len(migrated_files)}

## Deprecated Packages
The following packages have been deprecated and replaced with ipfs_kit_py:
"""
        
        for pkg in self.deprecated_packages:
            report += f"- {pkg}\n"
        
        report += f"""
## Files Modified
The following files were updated to use the new ipfs_kit_py API:
"""
        
        for file_path in migrated_files:
            relative_path = file_path.relative_to(self.project_root)
            report += f"- {relative_path}\n"
        
        report += f"""
## Next Steps
1. Test the migrated code to ensure functionality is maintained
2. Remove deprecated package dependencies from requirements.txt
3. Update documentation to reflect new API usage
4. Run comprehensive tests to validate the migration

## Rollback Instructions
If issues occur, restore from backup:
```bash
cp {self.backup_dir}/requirements.txt.backup requirements.txt
# Restore individual files from git if needed
git checkout -- <file_path>
```
"""
        
        report_file = self.project_root / "docs" / "IPFS_MIGRATION_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nMigration report saved to: {report_file}")

    def run_migration(self):
        """Execute the complete migration process."""
        print("Starting IPFS Kit migration...")
        print("=" * 50)
        
        # Step 1: Create backup
        print("Step 1: Creating backup...")
        self.create_backup()
        
        # Step 2: Install local ipfs_kit_py
        print("\nStep 2: Installing local ipfs_kit_py...")
        if not self.install_local_ipfs_kit():
            print("Failed to install local ipfs_kit_py. Aborting migration.")
            return False
        
        # Step 3: Update requirements.txt
        print("\nStep 3: Updating requirements.txt...")
        self.update_requirements()
        
        # Step 4: Migrate Python files
        print("\nStep 4: Migrating Python files...")
        migrated_files = self.migrate_python_files()
        
        # Step 5: Update test mocks
        print("\nStep 5: Updating test configuration...")
        self.update_conftest_mocks()
        
        # Step 6: Create migration report
        print("\nStep 6: Creating migration report...")
        self.create_migration_report(migrated_files)
        
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print(f"Backup created at: {self.backup_dir}")
        print(f"Files migrated: {len(migrated_files)}")
        print("\nNext steps:")
        print("1. Run tests to validate the migration")
        print("2. Review the migration report")
        print("3. Update any remaining manual references")
        
        return True

def main():
    project_root = "/home/barberb/laion-embeddings-1"
    migrator = IPFSKitMigrator(project_root)
    
    # Run the migration
    success = migrator.run_migration()
    
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed. Check the errors above.")

if __name__ == "__main__":
    main()
