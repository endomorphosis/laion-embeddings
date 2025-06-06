#!/usr/bin/env python3
"""
LAION MCP Server - Final Status Check
Comprehensive validation of the entire MCP server implementation
"""

import sys
import os
import json
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, '/home/barberb/laion-embeddings-1')

class MCPServerStatusChecker:
    def __init__(self):
        self.project_root = Path('/home/barberb/laion-embeddings-1')
        self.results = {}
    
    def check_file_structure(self):
        """Check if all required files exist"""
        print("📁 Checking File Structure...")
        
        required_files = [
            'mcp_server_minimal.py',
            'src/mcp_server/server.py',
            'src/mcp_server/tool_registry.py',
            'src/mcp_server/error_handlers.py',
            'src/mcp_server/validators.py',
            'src/mcp_server/fastapi_integration.py',
            'src/mcp_server/tools/embedding_tools.py',
            'src/mcp_server/tools/search_tools.py',
            'src/mcp_server/tools/storage_tools.py',
            'src/mcp_server/tools/analysis_tools.py',
            'src/mcp_server/tools/data_processing_tools.py',
            '.vscode/mcp.json'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - MISSING")
                missing_files.append(file_path)
        
        self.results['file_structure'] = {
            'status': 'PASS' if not missing_files else 'FAIL',
            'missing_files': missing_files,
            'total_files': len(required_files),
            'found_files': len(required_files) - len(missing_files)
        }
        
        return len(missing_files) == 0
    
    def check_imports(self):
        """Check if all modules can be imported"""
        print("\n📦 Checking Imports...")
        
        import_tests = [
            ('mcp_server_minimal', 'MinimalMCPServer'),
            ('src.mcp_server.server', 'MCPServer'),
            ('src.mcp_server.tool_registry', 'ToolRegistry'),
            ('src.mcp_server.error_handlers', 'handle_tool_error'),
            ('src.mcp_server.validators', 'ParameterValidator'),
        ]
        
        failed_imports = []
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"   ✅ {module_name}.{class_name}")
            except Exception as e:
                print(f"   ❌ {module_name}.{class_name} - {str(e)}")
                failed_imports.append((module_name, class_name, str(e)))
        
        self.results['imports'] = {
            'status': 'PASS' if not failed_imports else 'FAIL',
            'failed_imports': failed_imports,
            'total_tests': len(import_tests),
            'passed_tests': len(import_tests) - len(failed_imports)
        }
        
        return len(failed_imports) == 0
    
    def check_mcp_server_functionality(self):
        """Check MCP server core functionality"""
        print("\n🔧 Checking MCP Server Functionality...")
        
        try:
            from mcp_server_minimal import MinimalMCPServer
            server = MinimalMCPServer()
            
            # Check tools setup
            tools = server.tools
            expected_tools = ['generate_embedding', 'semantic_search', 'cluster_analysis', 'storage_management']
            
            found_tools = []
            missing_tools = []
            
            for tool in expected_tools:
                if tool in tools:
                    found_tools.append(tool)
                    print(f"   ✅ Tool: {tool}")
                else:
                    missing_tools.append(tool)
                    print(f"   ❌ Tool missing: {tool}")
            
            self.results['mcp_functionality'] = {
                'status': 'PASS' if not missing_tools else 'FAIL',
                'found_tools': found_tools,
                'missing_tools': missing_tools,
                'total_expected': len(expected_tools)
            }
            
            return len(missing_tools) == 0
            
        except Exception as e:
            print(f"   ❌ Server instantiation failed: {e}")
            self.results['mcp_functionality'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def check_vs_code_configuration(self):
        """Check VS Code MCP configuration"""
        print("\n⚙️  Checking VS Code Configuration...")
        
        try:
            config_path = self.project_root / '.vscode' / 'mcp.json'
            
            if not config_path.exists():
                print("   ❌ MCP configuration file missing")
                self.results['vs_code_config'] = {'status': 'FAIL', 'error': 'Config file missing'}
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'mcpServers' not in config:
                print("   ❌ No MCP servers configured")
                self.results['vs_code_config'] = {'status': 'FAIL', 'error': 'No servers configured'}
                return False
            
            servers = config['mcpServers']
            print(f"   ✅ Found {len(servers)} configured server(s)")
            
            for name, server_config in servers.items():
                print(f"   📡 Server: {name}")
                
                # Check if server file exists
                if 'args' in server_config and server_config['args']:
                    server_file = Path(server_config['args'][0])
                    if server_file.exists():
                        print(f"      ✅ Server file exists: {server_file}")
                    else:
                        print(f"      ❌ Server file missing: {server_file}")
                        self.results['vs_code_config'] = {'status': 'FAIL', 'error': f'Server file missing: {server_file}'}
                        return False
            
            self.results['vs_code_config'] = {
                'status': 'PASS',
                'servers_count': len(servers),
                'servers': list(servers.keys())
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Configuration check failed: {e}")
            self.results['vs_code_config'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def check_tool_registry(self):
        """Check tool registry functionality"""
        print("\n🔨 Checking Tool Registry...")
        
        try:
            from src.mcp_server.tool_registry import ToolRegistry
            
            # Create registry instance
            registry = ToolRegistry()
            
            # Check if tools can be loaded
            tools = registry.get_all_tools()
            print(f"   ✅ Registry created with {len(tools)} tools")
            
            # Check specific tool categories
            categories = {}
            for tool_name, tool_instance in tools.items():
                category = getattr(tool_instance, 'category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(tool_name)
            
            for category, tool_list in categories.items():
                print(f"   📂 Category '{category}': {len(tool_list)} tools")
                for tool in tool_list:
                    print(f"      - {tool}")
            
            self.results['tool_registry'] = {
                'status': 'PASS',
                'total_tools': len(tools),
                'categories': categories
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Tool registry check failed: {e}")
            traceback.print_exc()
            self.results['tool_registry'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def generate_final_report(self):
        """Generate final status report"""
        print("\n" + "="*60)
        print("📊 FINAL MCP SERVER STATUS REPORT")
        print("="*60)
        
        all_passed = True
        
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASS':
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                all_passed = False
        
        print("\n" + "-"*60)
        
        if all_passed:
            print("🎉 ALL SYSTEMS GO!")
            print("\n✅ Your LAION MCP Server is fully configured and ready!")
            print("\n📋 Next Steps:")
            print("   1. Restart VS Code to load the MCP configuration")
            print("   2. Open Claude in VS Code")
            print("   3. Claude will automatically connect to your LAION embeddings server")
            print("   4. Test with commands like:")
            print("      - 'Generate an embedding for this text'")
            print("      - 'Perform semantic search for similar content'")
            print("      - 'Analyze clusters in my embeddings'")
            print("      - 'Manage my embedding storage'")
            
            print("\n🔧 Available Tools:")
            if 'tool_registry' in self.results and 'categories' in self.results['tool_registry']:
                for category, tools in self.results['tool_registry']['categories'].items():
                    print(f"   📂 {category.title()}:")
                    for tool in tools:
                        print(f"      - {tool}")
        else:
            print("❌ SOME ISSUES FOUND")
            print("\n🔧 Please review the failed checks above and fix any issues.")
        
        print("\n" + "="*60)
        
        return all_passed
    
    def run_complete_check(self):
        """Run all checks"""
        print("🚀 LAION MCP Server - Complete Status Check")
        print("="*60)
        
        checks = [
            self.check_file_structure,
            self.check_imports,
            self.check_mcp_server_functionality,
            self.check_vs_code_configuration,
            self.check_tool_registry
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                print(f"❌ Check failed with exception: {e}")
                traceback.print_exc()
        
        return self.generate_final_report()

if __name__ == "__main__":
    checker = MCPServerStatusChecker()
    success = checker.run_complete_check()
    sys.exit(0 if success else 1)
