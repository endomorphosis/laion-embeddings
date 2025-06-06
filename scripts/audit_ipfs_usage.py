#!/usr/bin/env python3
"""
IPFS/Storacha Code Audit Script
Scans the codebase to identify existing IPFS and Storacha implementations
that need to be replaced with ipfs_kit_py.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set

class IPFSAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            'ipfs_imports': [],
            'storacha_imports': [],
            'ipfs_function_calls': [],
            'storacha_function_calls': [],
            'config_files': [],
            'dependencies': [],
            'files_to_modify': set(),
            'files_to_delete': set()
        }
        
        # Patterns to search for
        self.ipfs_patterns = [
            r'import.*ipfs',
            r'from.*ipfs.*import',
            r'ipfs[_\.]',
            r'IPFS[A-Z]',
            r'\.add\(',
            r'\.get\(',
            r'\.pin\(',
            r'\.hash\(',
            r'ipfshttpclient',
            r'py-ipfs-api',
        ]
        
        self.storacha_patterns = [
            r'import.*storacha',
            r'from.*storacha.*import',
            r'storacha[_\.]',
            r'STORACHA',
            r'\.upload\(',
            r'\.store\(',
            r'w3storage',
            r'web3storage'
        ]
        
        # File extensions to scan
        self.extensions = ['.py', '.yaml', '.yml', '.json', '.toml', '.txt', '.md']
        
        # Directories to skip
        self.skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', 'backup_deprecated_code'}

    def scan_file(self, file_path: Path) -> Dict:
        """Scan a single file for IPFS/Storacha usage."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {'error': str(e)}
        
        file_results = {
            'ipfs_matches': [],
            'storacha_matches': [],
            'line_numbers': {}
        }
        
        lines = content.split('\n')
        
        # Check for IPFS patterns
        for i, line in enumerate(lines, 1):
            for pattern in self.ipfs_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    file_results['ipfs_matches'].append({
                        'line': i,
                        'content': line.strip(),
                        'pattern': pattern
                    })
                    
            for pattern in self.storacha_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    file_results['storacha_matches'].append({
                        'line': i,
                        'content': line.strip(),
                        'pattern': pattern
                    })
        
        return file_results

    def scan_directory(self) -> None:
        """Recursively scan the project directory."""
        for root, dirs, files in os.walk(self.project_root):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip files without target extensions
                if file_path.suffix not in self.extensions:
                    continue
                
                # Scan the file
                results = self.scan_file(file_path)
                
                if 'error' in results:
                    continue
                
                # Record results if matches found
                if results['ipfs_matches'] or results['storacha_matches']:
                    relative_path = file_path.relative_to(self.project_root)
                    
                    if results['ipfs_matches']:
                        self.results['ipfs_imports'].append({
                            'file': str(relative_path), 
                            'matches': results['ipfs_matches']
                        })
                        
                    if results['storacha_matches']:
                        self.results['storacha_imports'].append({
                            'file': str(relative_path), 
                            'matches': results['storacha_matches']
                        })
                    
                    self.results['files_to_modify'].add(str(relative_path))

    def scan_dependencies(self) -> None:
        """Scan for dependency files and IPFS-related dependencies."""
        dep_files = [
            'requirements.txt',
            'pyproject.toml',
            'setup.py',
            'Pipfile',
            'environment.yml'
        ]
        
        for dep_file in dep_files:
            file_path = self.project_root / dep_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Look for IPFS-related dependencies
                    ipfs_deps = re.findall(r'.*ipfs.*', content, re.IGNORECASE)
                    storacha_deps = re.findall(r'.*storacha.*|.*w3storage.*', content, re.IGNORECASE)
                    
                    if ipfs_deps or storacha_deps:
                        self.results['dependencies'].append({
                            'file': dep_file,
                            'ipfs_deps': ipfs_deps,
                            'storacha_deps': storacha_deps
                        })
                        
                except Exception as e:
                    print(f"Error reading {dep_file}: {e}")

    def generate_report(self) -> str:
        """Generate a detailed audit report."""
        report = []
        report.append("# IPFS/Storacha Code Audit Report")
        report.append(f"Generated for project: {self.project_root}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Files with IPFS usage: {len(self.results['ipfs_imports'])}")
        report.append(f"- Files with Storacha usage: {len(self.results['storacha_imports'])}")
        report.append(f"- Total files to modify: {len(self.results['files_to_modify'])}")
        report.append(f"- Dependency files found: {len(self.results['dependencies'])}")
        report.append("")
        
        # IPFS Usage
        if self.results['ipfs_imports']:
            report.append("## IPFS Usage Found")
            for item in self.results['ipfs_imports']:
                report.append(f"### {item['file']}")
                for match in item['matches']:
                    report.append(f"- Line {match['line']}: `{match['content']}`")
                report.append("")
        
        # Storacha Usage
        if self.results['storacha_imports']:
            report.append("## Storacha Usage Found")
            for item in self.results['storacha_imports']:
                report.append(f"### {item['file']}")
                for match in item['matches']:
                    report.append(f"- Line {match['line']}: `{match['content']}`")
                report.append("")
        
        # Dependencies
        if self.results['dependencies']:
            report.append("## Dependencies to Update")
            for dep in self.results['dependencies']:
                report.append(f"### {dep['file']}")
                if dep['ipfs_deps']:
                    report.append("IPFS dependencies:")
                    for d in dep['ipfs_deps']:
                        report.append(f"- {d}")
                if dep['storacha_deps']:
                    report.append("Storacha dependencies:")
                    for d in dep['storacha_deps']:
                        report.append(f"- {d}")
                report.append("")
        
        # Files to modify
        if self.results['files_to_modify']:
            report.append("## Files That Need Modification")
            for file in sorted(self.results['files_to_modify']):
                report.append(f"- {file}")
            report.append("")
        
        return '\n'.join(report)

    def save_results(self, output_file: str) -> None:
        """Save results to JSON file."""
        # Convert set to list for JSON serialization
        results_copy = self.results.copy()
        results_copy['files_to_modify'] = list(results_copy['files_to_modify'])
        results_copy['files_to_delete'] = list(results_copy['files_to_delete'])
        
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)

def main():
    project_root = "/home/barberb/laion-embeddings-1"
    auditor = IPFSAuditor(project_root)
    
    print("Starting IPFS/Storacha code audit...")
    auditor.scan_directory()
    auditor.scan_dependencies()
    
    # Generate and save report
    report = auditor.generate_report()
    
    # Save detailed results
    auditor.save_results(f"{project_root}/docs/ipfs_audit_results.json")
    
    # Save report
    with open(f"{project_root}/docs/IPFS_AUDIT_REPORT.md", 'w') as f:
        f.write(report)
    
    print("Audit complete!")
    print(f"Report saved to: {project_root}/docs/IPFS_AUDIT_REPORT.md")
    print(f"Detailed results saved to: {project_root}/docs/ipfs_audit_results.json")
    
    # Print summary
    print("\nSummary:")
    print(f"- Files with IPFS usage: {len(auditor.results['ipfs_imports'])}")
    print(f"- Files with Storacha usage: {len(auditor.results['storacha_imports'])}")
    print(f"- Total files to modify: {len(auditor.results['files_to_modify'])}")

if __name__ == "__main__":
    main()
