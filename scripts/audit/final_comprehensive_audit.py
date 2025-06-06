#!/usr/bin/env python3
"""
COMPREHENSIVE MCP AUDIT SCRIPT
============================
This script performs a complete audit of FastAPI endpoints vs MCP tools
to ensure comprehensive coverage and identify any gaps.
"""

import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def extract_fastapi_endpoints():
    """Extract all FastAPI endpoints from main.py"""
    endpoints = []
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        # Find all app decorators
        pattern = r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\'].*?\)\s*(?:async\s+)?def\s+(\w+)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for method, path, func_name in matches:
            endpoints.append({
                'method': method.upper(),
                'path': path,
                'function': func_name,
                'category': categorize_endpoint(path, func_name)
            })
    
    except Exception as e:
        print(f"Error reading main.py: {e}")
    
    return endpoints

def categorize_endpoint(path, func_name):
    """Categorize endpoint based on path and function name"""
    path_lower = path.lower()
    func_lower = func_name.lower()
    
    if any(x in path_lower for x in ['health', 'status']):
        return 'health'
    elif any(x in path_lower for x in ['embed', 'embedding']):
        return 'embedding'
    elif any(x in path_lower for x in ['search']):
        return 'search'
    elif any(x in path_lower for x in ['load', 'shard']):
        return 'index_management'
    elif any(x in path_lower for x in ['sparse']):
        return 'sparse_embedding'
    elif any(x in path_lower for x in ['ipfs', 'cluster', 'storacha']):
        return 'storage'
    elif any(x in path_lower for x in ['cache']):
        return 'cache'
    elif any(x in path_lower for x in ['auth', 'login']):
        return 'auth'
    elif any(x in path_lower for x in ['metric']):
        return 'monitoring'
    elif any(x in func_lower for x in ['admin', 'endpoint']):
        return 'admin'
    else:
        return 'other'

def scan_mcp_tools():
    """Scan all MCP tool files and extract tool information"""
    tools_dir = Path("src/mcp_server/tools")
    tools = []
    
    if not tools_dir.exists():
        print(f"Tools directory not found: {tools_dir}")
        return tools
    
    for tool_file in tools_dir.glob("*.py"):
        if tool_file.name.startswith("__"):
            continue
            
        try:
            with open(tool_file, "r") as f:
                content = f.read()
            
            # Find class definitions that inherit from ClaudeMCPTool
            class_pattern = r'class\s+(\w+)\(.*ClaudeMCPTool.*?\):'
            classes = re.findall(class_pattern, content)
            
            for class_name in classes:
                # Extract tool name and description
                tool_info = extract_tool_info(content, class_name)
                if tool_info:
                    tool_info['file'] = str(tool_file)
                    tool_info['class_name'] = class_name
                    tool_info['category'] = categorize_tool(tool_file.name, class_name)
                    tools.append(tool_info)
        
        except Exception as e:
            print(f"Error reading {tool_file}: {e}")
    
    return tools

def extract_tool_info(content, class_name):
    """Extract tool name and description from class content"""
    try:
        # Find the class definition
        class_pattern = rf'class\s+{class_name}\(.*?\):(.*?)(?=class|\Z)'
        class_match = re.search(class_pattern, content, re.DOTALL)
        
        if not class_match:
            return None
        
        class_content = class_match.group(1)
        
        # Extract tool name
        name_pattern = r'self\.name\s*=\s*["\']([^"\']+)["\']'
        name_match = re.search(name_pattern, class_content)
        tool_name = name_match.group(1) if name_match else class_name.lower()
        
        # Extract description
        desc_pattern = r'self\.description\s*=\s*["\']([^"\']+)["\']'
        desc_match = re.search(desc_pattern, class_content)
        description = desc_match.group(1) if desc_match else "No description found"
        
        return {
            'name': tool_name,
            'description': description
        }
    
    except Exception as e:
        print(f"Error extracting tool info for {class_name}: {e}")
        return None

def categorize_tool(filename, class_name):
    """Categorize tool based on filename and class name"""
    filename_lower = filename.lower()
    class_lower = class_name.lower()
    
    if 'embedding' in filename_lower:
        if 'sparse' in filename_lower:
            return 'sparse_embedding'
        else:
            return 'embedding'
    elif 'search' in filename_lower:
        return 'search'
    elif 'storage' in filename_lower:
        return 'storage'
    elif 'ipfs' in filename_lower or 'cluster' in filename_lower:
        return 'storage'
    elif 'analysis' in filename_lower:
        return 'analysis'
    elif 'vector' in filename_lower:
        return 'vector_store'
    elif 'cache' in filename_lower:
        return 'cache'
    elif 'auth' in filename_lower:
        return 'auth'
    elif 'monitoring' in filename_lower:
        return 'monitoring'
    elif 'admin' in filename_lower:
        return 'admin'
    elif 'session' in filename_lower:
        return 'session'
    elif 'index' in filename_lower:
        return 'index_management'
    elif 'workflow' in filename_lower:
        return 'workflow'
    else:
        return 'other'

def check_mcp_registration():
    """Check which tools are actually registered in the MCP server"""
    registered_tools = []
    
    try:
        with open("src/mcp_server/main.py", "r") as f:
            content = f.read()
        
        # Find import statements for tools
        import_pattern = r'from\s+\.tools\.(\w+)\s+import\s+([^)]+)'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        
        for module, classes in imports:
            # Clean up class names
            class_names = [c.strip() for c in classes.split(',')]
            for class_name in class_names:
                registered_tools.append({
                    'module': module,
                    'class': class_name.strip(),
                    'category': categorize_tool(f"{module}.py", class_name)
                })
    
    except Exception as e:
        print(f"Error checking MCP registration: {e}")
    
    return registered_tools

def analyze_coverage(endpoints, tools, registered):
    """Analyze coverage between endpoints and tools"""
    analysis = {
        'total_endpoints': len(endpoints),
        'total_tools': len(tools),
        'registered_tools': len(registered),
        'coverage_by_category': defaultdict(lambda: {'endpoints': 0, 'tools': 0, 'registered': 0}),
        'gaps': {'missing_tools': [], 'unregistered_tools': [], 'uncovered_endpoints': []},
        'recommendations': []
    }
    
    # Group by category
    endpoint_categories = defaultdict(list)
    tool_categories = defaultdict(list)
    registered_categories = defaultdict(list)
    
    for ep in endpoints:
        endpoint_categories[ep['category']].append(ep)
        analysis['coverage_by_category'][ep['category']]['endpoints'] += 1
    
    for tool in tools:
        tool_categories[tool['category']].append(tool)
        analysis['coverage_by_category'][tool['category']]['tools'] += 1
    
    for reg in registered:
        registered_categories[reg['category']].append(reg)
        analysis['coverage_by_category'][reg['category']]['registered'] += 1
    
    # Find gaps
    all_categories = set(endpoint_categories.keys()) | set(tool_categories.keys())
    
    for category in all_categories:
        ep_count = len(endpoint_categories[category])
        tool_count = len(tool_categories[category])
        reg_count = len(registered_categories[category])
        
        if ep_count > 0 and tool_count == 0:
            analysis['gaps']['missing_tools'].append(category)
        
        if tool_count > reg_count:
            analysis['gaps']['unregistered_tools'].append(category)
        
        if ep_count > 0 and reg_count == 0:
            analysis['gaps']['uncovered_endpoints'].append(category)
    
    # Generate recommendations
    if analysis['gaps']['missing_tools']:
        analysis['recommendations'].append(f"Create tools for: {', '.join(analysis['gaps']['missing_tools'])}")
    
    if analysis['gaps']['unregistered_tools']:
        analysis['recommendations'].append(f"Register tools for: {', '.join(analysis['gaps']['unregistered_tools'])}")
    
    if analysis['gaps']['uncovered_endpoints']:
        analysis['recommendations'].append(f"Ensure endpoints are covered: {', '.join(analysis['gaps']['uncovered_endpoints'])}")
    
    return analysis

def generate_report(endpoints, tools, registered, analysis):
    """Generate comprehensive audit report"""
    print("🚀 MCP COMPREHENSIVE AUDIT RESULTS")
    print("=" * 60)
    
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"  FastAPI Endpoints: {analysis['total_endpoints']}")
    print(f"  Available Tools:   {analysis['total_tools']}")
    print(f"  Registered Tools:  {analysis['registered_tools']}")
    
    print(f"\n📋 ENDPOINT BREAKDOWN BY CATEGORY")
    for category, data in analysis['coverage_by_category'].items():
        print(f"  {category:15} | EP:{data['endpoints']:2} | Tools:{data['tools']:2} | Reg:{data['registered']:2}")
    
    print(f"\n🔍 DETAILED ENDPOINT ANALYSIS")
    for ep in sorted(endpoints, key=lambda x: x['category']):
        print(f"  {ep['method']:6} {ep['path']:25} -> {ep['function']:20} [{ep['category']}]")
    
    print(f"\n🛠️  AVAILABLE TOOLS")
    for tool in sorted(tools, key=lambda x: x['category']):
        print(f"  {tool['name']:30} [{tool['category']}] - {tool['description'][:50]}...")
    
    print(f"\n✅ REGISTERED TOOLS")
    for reg in sorted(registered, key=lambda x: x['category']):
        print(f"  {reg['class']:30} [{reg['category']}] from {reg['module']}")
    
    print(f"\n⚠️  IDENTIFIED GAPS")
    if analysis['gaps']['missing_tools']:
        print(f"  Missing tools for: {', '.join(analysis['gaps']['missing_tools'])}")
    if analysis['gaps']['unregistered_tools']:
        print(f"  Unregistered tools: {', '.join(analysis['gaps']['unregistered_tools'])}")
    if analysis['gaps']['uncovered_endpoints']:
        print(f"  Uncovered endpoints: {', '.join(analysis['gaps']['uncovered_endpoints'])}")
    
    print(f"\n💡 RECOMMENDATIONS")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Calculate coverage score
    total_categories = len(set(ep['category'] for ep in endpoints))
    covered_categories = len(set(ep['category'] for ep in endpoints) & set(reg['category'] for reg in registered))
    coverage_score = (covered_categories / total_categories * 100) if total_categories > 0 else 0
    
    print(f"\n🎯 OVERALL COVERAGE SCORE: {coverage_score:.1f}%")
    
    if coverage_score >= 90:
        print("✅ EXCELLENT - Comprehensive tool coverage achieved!")
    elif coverage_score >= 75:
        print("⚠️  GOOD - Minor gaps to address")
    elif coverage_score >= 50:
        print("🚧 MODERATE - Significant improvements needed")
    else:
        print("❌ POOR - Major coverage gaps require immediate attention")

def save_results(endpoints, tools, registered, analysis):
    """Save results to files"""
    # Save detailed results as JSON
    results = {
        'audit_timestamp': str(Path(__file__).stat().st_mtime),
        'summary': {
            'total_endpoints': analysis['total_endpoints'],
            'total_tools': analysis['total_tools'],
            'registered_tools': analysis['registered_tools']
        },
        'endpoints': endpoints,
        'tools': tools,
        'registered': registered,
        'analysis': dict(analysis)  # Convert defaultdict to dict
    }
    
    with open("MCP_AUDIT_COMPREHENSIVE.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save markdown summary
    coverage_score = 0
    if analysis['total_endpoints'] > 0:
        total_categories = len(set(ep['category'] for ep in endpoints))
        covered_categories = len(set(ep['category'] for ep in endpoints) & set(reg['category'] for reg in registered))
        coverage_score = (covered_categories / total_categories * 100) if total_categories > 0 else 0
    
    with open("MCP_AUDIT_SUMMARY.md", "w") as f:
        f.write("# MCP Comprehensive Audit Summary\n\n")
        f.write(f"**Coverage Score:** {coverage_score:.1f}%\n")
        f.write(f"**FastAPI Endpoints:** {analysis['total_endpoints']}\n")
        f.write(f"**Available MCP Tools:** {analysis['total_tools']}\n")
        f.write(f"**Registered Tools:** {analysis['registered_tools']}\n\n")
        
        f.write("## Key Findings\n")
        if analysis['gaps']['missing_tools']:
            f.write(f"- **Missing Tools:** {', '.join(analysis['gaps']['missing_tools'])}\n")
        if analysis['gaps']['unregistered_tools']:
            f.write(f"- **Unregistered Tools:** {', '.join(analysis['gaps']['unregistered_tools'])}\n")
        if analysis['gaps']['uncovered_endpoints']:
            f.write(f"- **Uncovered Endpoints:** {', '.join(analysis['gaps']['uncovered_endpoints'])}\n")
        
        f.write("\n## Recommendations\n")
        for rec in analysis['recommendations']:
            f.write(f"- {rec}\n")

def main():
    """Run comprehensive audit"""
    print("Starting comprehensive MCP audit...")
    
    # Extract all data
    endpoints = extract_fastapi_endpoints()
    tools = scan_mcp_tools()
    registered = check_mcp_registration()
    
    # Analyze coverage
    analysis = analyze_coverage(endpoints, tools, registered)
    
    # Generate report
    generate_report(endpoints, tools, registered, analysis)
    
    # Save results
    save_results(endpoints, tools, registered, analysis)
    
    print(f"\n📁 Results saved to:")
    print(f"  - MCP_AUDIT_COMPREHENSIVE.json (detailed data)")
    print(f"  - MCP_AUDIT_SUMMARY.md (summary report)")

if __name__ == "__main__":
    main()
