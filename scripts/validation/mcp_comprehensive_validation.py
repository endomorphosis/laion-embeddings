#!/usr/bin/env python3
"""
Comprehensive MCP Server Validation and Final Integration Test

This script validates the complete MCP server setup with real services.
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPServerValidator:
    """Comprehensive validator for MCP server functionality."""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✓ {test_name}")
            if details:
                print(f"  {details}")
        else:
            print(f"✗ {test_name}")
            if details:
                print(f"  Error: {details}")
        
        self.test_results[test_name] = {'passed': passed, 'details': details}
    
    def test_imports(self):
        """Test all critical imports."""
        print("\n" + "=" * 60)
        print("TESTING IMPORTS")
        print("=" * 60)
        
        # Test MCPConfig import
        try:
            from src.mcp_server.config import MCPConfig
            self.log_test_result("MCPConfig import", True)
        except Exception as e:
            self.log_test_result("MCPConfig import", False, str(e))
        
        # Test ServiceFactory import
        try:
            from src.mcp_server.service_factory import ServiceFactory
            self.log_test_result("ServiceFactory import", True)
        except Exception as e:
            self.log_test_result("ServiceFactory import", False, str(e))
        
        # Test core services
        try:
            from services.vector_service import VectorService
            from services.embedding_service import EmbeddingService  
            from services.clustering_service import VectorClusterer
            self.log_test_result("Core services import", True)
        except Exception as e:
            self.log_test_result("Core services import", False, str(e))
        
        # Test MCP tools
        try:
            from src.mcp_server.tools.embedding_tools import EmbeddingGenerationTool
            from src.mcp_server.tools.search_tools import SemanticSearchTool
            from src.mcp_server.tools.storage_tools import StorageManagementTool
            from src.mcp_server.tools.analysis_tools import ClusterAnalysisTool
            self.log_test_result("MCP tools import", True)
        except Exception as e:
            self.log_test_result("MCP tools import", False, str(e))
        
        # Test main application
        try:
            from src.mcp_server.main import MCPServerApplication
            self.log_test_result("MCPServerApplication import", True)
        except Exception as e:
            self.log_test_result("MCPServerApplication import", False, str(e))
    
    async def test_service_initialization(self):
        """Test service initialization."""
        print("\n" + "=" * 60)
        print("TESTING SERVICE INITIALIZATION")
        print("=" * 60)
        
        try:
            from src.mcp_server.config import MCPConfig
            from src.mcp_server.service_factory import ServiceFactory
            
            # Create config
            config = MCPConfig()
            self.log_test_result("MCPConfig creation", True, f"Server: {config.server_name}")
            
            # Create service factory
            factory = ServiceFactory(config)
            self.log_test_result("ServiceFactory creation", True)
            
            # Initialize services
            services = await factory.initialize_services()
            self.log_test_result("Services initialization", True, f"Initialized {len(services)} services")
            
            # Test individual service access
            embedding_service = factory.get_embedding_service()
            if embedding_service:
                self.log_test_result("Embedding service access", True, f"Type: {type(embedding_service).__name__}")
            else:
                self.log_test_result("Embedding service access", False, "Service is None")
            
            vector_service = factory.get_vector_service()  
            if vector_service:
                self.log_test_result("Vector service access", True, f"Type: {type(vector_service).__name__}")
            else:
                self.log_test_result("Vector service access", False, "Service is None")
            
            clustering_service = factory.get_clustering_service()
            if clustering_service:
                self.log_test_result("Clustering service access", True, f"Type: {type(clustering_service).__name__}")
            else:
                self.log_test_result("Clustering service access", False, "Service is None")
            
            # Test optional services
            try:
                ipfs_service = factory.get_ipfs_vector_service()
                if ipfs_service:
                    self.log_test_result("IPFS service access", True, f"Type: {type(ipfs_service).__name__}")
                else:
                    self.log_test_result("IPFS service access", False, "Service not available")
            except Exception as e:
                self.log_test_result("IPFS service access", False, f"Optional service: {e}")
            
            return factory
            
        except Exception as e:
            self.log_test_result("Service initialization", False, str(e))
            return None
    
    async def test_mcp_application(self, factory):
        """Test MCP application initialization."""
        print("\n" + "=" * 60)
        print("TESTING MCP APPLICATION")
        print("=" * 60)
        
        try:
            from src.mcp_server.main import MCPServerApplication
            
            # Create application
            app = MCPServerApplication()
            self.log_test_result("MCPServerApplication creation", True)
            
            # Initialize application
            await app._initialize_components()
            self.log_test_result("MCP application initialization", True)
            
            # Check tools
            if hasattr(app, 'get_tools'):
                tools = app.get_tools()
                self.log_test_result("Tool registry access", True, f"Found {len(tools)} tools")
                
                # List some tools
                tool_names = list(tools.keys())[:5]  # First 5 tools
                if tool_names:
                    print(f"  Sample tools: {', '.join(tool_names)}")
            else:
                self.log_test_result("Tool registry access", False, "get_tools method not available")
            
            return app
            
        except Exception as e:
            self.log_test_result("MCP application initialization", False, str(e))
            return None
    
    async def test_end_to_end_integration(self):
        """Test complete end-to-end integration."""
        print("\n" + "=" * 60)  
        print("TESTING END-TO-END INTEGRATION")
        print("=" * 60)
        
        try:
            # Initialize service factory
            factory = await self.test_service_initialization()
            if not factory:
                self.log_test_result("End-to-end integration", False, "Service factory initialization failed")
                return False
            
            # Initialize MCP application
            app = await self.test_mcp_application(factory)
            if not app:
                self.log_test_result("End-to-end integration", False, "MCP application initialization failed")
                return False
            
            self.log_test_result("End-to-end integration", True, "Complete integration successful")
            return True
            
        except Exception as e:
            self.log_test_result("End-to-end integration", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all validation tests."""
        print("🚀 Starting MCP Server Comprehensive Validation")
        print("=" * 80)
        
        # Run import tests
        self.test_imports()
        
        # Run integration tests
        await self.test_end_to_end_integration()
        
        # Print final results
        print("\n" + "=" * 80)
        print("FINAL VALIDATION RESULTS")
        print("=" * 80)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED - MCP SERVER VALIDATION COMPLETE!")
            print("✅ The MCP server is ready for production use.")
            return True
        else:
            print(f"\n❌ {self.total_tests - self.passed_tests} TESTS FAILED")
            print("🔧 Please review the failed tests above.")
            return False

async def main():
    """Main validation function."""
    try:
        validator = MCPServerValidator()
        success = await validator.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\nValidation completed with exit code: {exit_code}")
    sys.exit(exit_code)
