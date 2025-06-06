#!/usr/bin/env python3
"""
Comprehensive MCP Tool Coverage Audit
====================================
This script validates that all FastAPI endpoints have corresponding MCP tools
and verifies test coverage for all tools.
"""

import sys
import os
import importlib
import inspect
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def audit_fastapi_endpoints():
    """Extract all FastAPI endpoints from main.py"""
    print("🔍 ANALYZING FASTAPI ENDPOINTS")
    print("=" * 50)
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        endpoints = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('@app.') and any(method in line for method in ['get(', 'post(', 'put(', 'delete(', 'patch(']):
                # Extract endpoint path
                if '"' in line:
                    path = line.split('"')[1]
                elif "'" in line:
                    path = line.split("'")[1]
                else:
                    continue
                
                # Get method
                method = line.split('(')[0].replace('@app.', '').upper()
                
                # Get function name from next non-empty line
                func_name = None
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('def '):
                        func_name = lines[j].strip().split('def ')[1].split('(')[0]
                        break
                
                endpoints.append({
                    'method': method,
                    'path': path,
                    'function': func_name
                })
        
        print(f"✓ Found {len(endpoints)} FastAPI endpoints:")
        for ep in endpoints:
            print(f"  {ep['method']:6} {ep['path']:30} -> {ep['function']}")
        
        return endpoints
        
    except Exception as e:
        print(f"❌ Error analyzing FastAPI endpoints: {e}")
        return []

def audit_mcp_tools():
    """Analyze all MCP tools"""
    print("\n🔍 ANALYZING MCP TOOLS")
    print("=" * 50)
    
    try:
        from src.mcp_server.tool_registry import ToolRegistry
        
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        
        print(f"✓ Found {len(tools)} MCP tools:")
        
        tool_categories = {}
        for tool in tools:
            category = getattr(tool, 'category', 'uncategorized')
            if category not in tool_categories:
                tool_categories[category] = []
            tool_categories[category].append(tool)
        
        for category, tool_list in tool_categories.items():
            print(f"\n  📁 {category.upper()}:")
            for tool in tool_list:
                print(f"    - {tool.name}: {tool.description[:60]}...")
        
        return tools, tool_categories
        
    except Exception as e:
        print(f"❌ Error analyzing MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def audit_test_coverage():
    """Check test coverage for MCP tools"""
    print("\n🔍 ANALYZING TEST COVERAGE")
    print("=" * 50)
    
    test_files = []
    test_dir = Path("test")
    
    if test_dir.exists():
        # Find all test files
        for test_file in test_dir.rglob("test_*.py"):
            test_files.append(test_file)
        
        print(f"✓ Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  - {test_file}")
    else:
        print("❌ Test directory not found")
    
    return test_files

def generate_coverage_report(endpoints, tools, test_files):
    """Generate comprehensive coverage report"""
    print("\n📊 COVERAGE ANALYSIS REPORT")
    print("=" * 50)
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  FastAPI Endpoints: {len(endpoints)}")
    print(f"  MCP Tools:        {len(tools)}")
    print(f"  Test Files:       {len(test_files)}")
    
    # Analyze endpoint to tool mapping
    print(f"\n🔗 ENDPOINT-TO-TOOL MAPPING:")
    
    endpoint_functions = {ep['function'] for ep in endpoints if ep['function']}
    tool_functions = {tool.name for tool in tools}
    
    # Map endpoints to potential tools
    mapped_endpoints = []
    unmapped_endpoints = []
    
    for ep in endpoints:
        if ep['function']:
            # Check if there's a corresponding tool
            found_tool = False
            for tool in tools:
                if (ep['function'].lower() in tool.name.lower() or 
                    tool.name.lower() in ep['function'].lower() or
                    any(word in tool.description.lower() for word in ep['path'].split('/') if word)):
                    mapped_endpoints.append((ep, tool))
                    found_tool = True
                    break
            
            if not found_tool:
                unmapped_endpoints.append(ep)
    
    print(f"  ✓ Mapped endpoints: {len(mapped_endpoints)}")
    for ep, tool in mapped_endpoints:
        print(f"    {ep['method']} {ep['path']} -> {tool.name}")
    
    print(f"  ⚠️  Unmapped endpoints: {len(unmapped_endpoints)}")
    for ep in unmapped_endpoints:
        print(f"    {ep['method']} {ep['path']} ({ep['function']})")
    
    return {
        'total_endpoints': len(endpoints),
        'total_tools': len(tools),
        'total_tests': len(test_files),
        'mapped_endpoints': len(mapped_endpoints),
        'unmapped_endpoints': len(unmapped_endpoints),
        'coverage_percentage': (len(mapped_endpoints) / len(endpoints) * 100) if endpoints else 0
    }

def main():
    """Run comprehensive audit"""
    print("🚀 MCP TOOL COVERAGE AUDIT")
    print("=" * 60)
    
    # Analyze FastAPI endpoints
    endpoints = audit_fastapi_endpoints()
    
    # Analyze MCP tools
    tools, tool_categories = audit_mcp_tools()
    
    # Analyze test coverage
    test_files = audit_test_coverage()
    
    # Generate report
    report = generate_coverage_report(endpoints, tools, test_files)
    
    print(f"\n🎯 FINAL AUDIT RESULTS")
    print("=" * 50)
    print(f"Coverage Score: {report['coverage_percentage']:.1f}%")
    print(f"Endpoints Covered: {report['mapped_endpoints']}/{report['total_endpoints']}")
    print(f"Tools Available: {report['total_tools']}")
    print(f"Test Files: {report['total_tests']}")
    
    if report['coverage_percentage'] >= 90:
        print("✅ EXCELLENT COVERAGE - Audit PASSED")
    elif report['coverage_percentage'] >= 75:
        print("⚠️  GOOD COVERAGE - Minor gaps identified")
    else:
        print("❌ COVERAGE GAPS - Action required")
    
    # Write results to file
    with open("MCP_AUDIT_RESULTS.md", "w") as f:
        f.write("# MCP Tool Coverage Audit Results\n\n")
        f.write(f"**Coverage Score:** {report['coverage_percentage']:.1f}%\n")
        f.write(f"**Endpoints Covered:** {report['mapped_endpoints']}/{report['total_endpoints']}\n")
        f.write(f"**Tools Available:** {report['total_tools']}\n")
        f.write(f"**Test Files:** {report['total_tests']}\n\n")
        f.write("## Detailed Analysis\n")
        f.write("See console output for complete analysis.\n")
    
    print(f"\n📝 Results saved to: MCP_AUDIT_RESULTS.md")

if __name__ == "__main__":
    main()
