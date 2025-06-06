"""
Comprehensive tests for workflow orchestration MCP tools.
"""

import pytest
import asyncio
import tempfile
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Import the workflow tools
from src.mcp_server.tools.workflow_tools import (
    execute_workflow_tool,
    create_embedding_pipeline_tool,
    get_workflow_status_tool,
    list_workflows_tool
)


class TestExecuteWorkflowTool:
    """Test execute_workflow_tool function."""
    
    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, temp_dir):
        """Test execution of a simple workflow."""
        workflow_config = {
            "name": "simple_embedding_workflow",
            "steps": [
                {
                    "name": "load_data",
                    "type": "data_loader",
                    "parameters": {
                        "source": "test_data.jsonl",
                        "format": "jsonl"
                    }
                },
                {
                    "name": "create_embeddings",
                    "type": "embedding_generator",
                    "parameters": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "batch_size": 32
                    }
                }
            ]
        }
        
        result = await execute_workflow_tool(
            workflow_config=workflow_config,
            execution_context={"workspace": temp_dir}
        )
        
        assert result["success"] is True
        assert "workflow_id" in result
        assert "execution_id" in result
        assert result["status"] in ["completed", "running", "pending"]
        assert "start_time" in result
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_dependencies(self, temp_dir):
        """Test execution of workflow with step dependencies."""
        workflow_config = {
            "name": "dependent_workflow",
            "steps": [
                {
                    "name": "load_data",
                    "type": "data_loader",
                    "parameters": {"source": "test.csv"}
                },
                {
                    "name": "preprocess",
                    "type": "data_processor",
                    "depends_on": ["load_data"],
                    "parameters": {"clean_text": True}
                },
                {
                    "name": "embed",
                    "type": "embedding_generator",
                    "depends_on": ["preprocess"],
                    "parameters": {"model": "test-model"}
                },
                {
                    "name": "store",
                    "type": "vector_store",
                    "depends_on": ["embed"],
                    "parameters": {"store_type": "faiss"}
                }
            ]
        }
        
        result = await execute_workflow_tool(
            workflow_config=workflow_config,
            execution_context={"workspace": temp_dir}
        )
        
        assert result["success"] is True
        assert "execution_plan" in result
        assert len(result["execution_plan"]) == 4  # All steps planned
    
    @pytest.mark.asyncio
    async def test_execute_workflow_async(self, temp_dir):
        """Test asynchronous workflow execution."""
        workflow_config = {
            "name": "async_workflow",
            "execution_mode": "async",
            "steps": [
                {
                    "name": "long_running_task",
                    "type": "batch_processor",
                    "parameters": {"batch_size": 1000, "delay": 0.1}
                }
            ]
        }
        
        result = await execute_workflow_tool(
            workflow_config=workflow_config,
            async_execution=True
        )
        
        assert result["success"] is True
        assert result["status"] == "running"
        assert "workflow_id" in result
        assert "execution_id" in result
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_parameters(self, temp_dir):
        """Test workflow execution with runtime parameters."""
        workflow_config = {
            "name": "parameterized_workflow",
            "parameters": {
                "input_file": {"type": "string", "required": True},
                "model_name": {"type": "string", "default": "default-model"},
                "batch_size": {"type": "integer", "default": 32}
            },
            "steps": [
                {
                    "name": "process",
                    "type": "processor",
                    "parameters": {
                        "input": "{{input_file}}",
                        "model": "{{model_name}}",
                        "batch_size": "{{batch_size}}"
                    }
                }
            ]
        }
        
        runtime_params = {
            "input_file": "test_input.jsonl",
            "model_name": "custom-model",
            "batch_size": 64
        }
        
        result = await execute_workflow_tool(
            workflow_config=workflow_config,
            parameters=runtime_params
        )
        
        assert result["success"] is True
        assert result["parameters"] == runtime_params
    
    @pytest.mark.asyncio
    async def test_execute_workflow_invalid_config(self):
        """Test workflow execution with invalid configuration."""
        invalid_config = {
            "name": "invalid_workflow",
            "steps": []  # Empty steps
        }
        
        result = await execute_workflow_tool(workflow_config=invalid_config)
        
        assert result["success"] is False
        assert "error" in result
        assert "steps" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_workflow_circular_dependency(self):
        """Test workflow execution with circular dependencies."""
        workflow_config = {
            "name": "circular_workflow",
            "steps": [
                {
                    "name": "step_a",
                    "type": "processor",
                    "depends_on": ["step_b"]
                },
                {
                    "name": "step_b",
                    "type": "processor",
                    "depends_on": ["step_a"]
                }
            ]
        }
        
        result = await execute_workflow_tool(workflow_config=workflow_config)
        
        assert result["success"] is False
        assert "error" in result
        assert "circular" in result["error"].lower()


class TestCreateEmbeddingPipelineTool:
    """Test create_embedding_pipeline_tool function."""
    
    @pytest.mark.asyncio
    async def test_create_basic_pipeline(self, temp_dir):
        """Test creation of basic embedding pipeline."""
        pipeline_config = {
            "name": "basic_embedding_pipeline",
            "input_source": "data/input.jsonl",
            "output_destination": "data/embeddings.npz",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is True
        assert "pipeline_id" in result
        assert "workflow_config" in result
        assert result["pipeline_type"] == "embedding"
        assert "estimated_duration" in result
    
    @pytest.mark.asyncio
    async def test_create_pipeline_with_preprocessing(self, temp_dir):
        """Test creation of pipeline with preprocessing steps."""
        pipeline_config = {
            "name": "preprocessing_pipeline",
            "input_source": "raw_data.csv",
            "preprocessing": {
                "text_cleaning": True,
                "tokenization": True,
                "chunking": {"strategy": "sentence", "max_length": 512}
            },
            "model": "openai/text-embedding-ada-002",
            "batch_size": 100
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is True
        assert "preprocessing_steps" in result["workflow_config"]
        assert len(result["workflow_config"]["steps"]) > 2  # Multiple steps
    
    @pytest.mark.asyncio
    async def test_create_pipeline_with_vector_store(self, temp_dir):
        """Test creation of pipeline with vector store integration."""
        pipeline_config = {
            "name": "store_pipeline",
            "input_source": "documents.jsonl",
            "model": "sentence-transformers/all-mpnet-base-v2",
            "vector_store": {
                "type": "faiss",
                "index_type": "flat",
                "dimension": 768
            },
            "storage_config": {
                "store_embeddings": True,
                "store_metadata": True
            }
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is True
        assert "vector_store_config" in result
        assert result["vector_store_config"]["type"] == "faiss"
    
    @pytest.mark.asyncio
    async def test_create_pipeline_with_sharding(self, temp_dir):
        """Test creation of pipeline with sharding configuration."""
        pipeline_config = {
            "name": "sharded_pipeline",
            "input_source": "large_dataset.jsonl",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "sharding": {
                "enabled": True,
                "shard_size": 10000,
                "max_shards": 10
            },
            "parallel_processing": True
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is True
        assert "sharding_config" in result
        assert result["sharding_config"]["enabled"] is True
        assert "parallel_steps" in result["workflow_config"]
    
    @pytest.mark.asyncio
    async def test_create_pipeline_invalid_model(self, temp_dir):
        """Test pipeline creation with invalid model."""
        pipeline_config = {
            "name": "invalid_model_pipeline",
            "input_source": "data.jsonl",
            "model": "nonexistent/model"
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "model" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_create_pipeline_missing_source(self, temp_dir):
        """Test pipeline creation with missing input source."""
        pipeline_config = {
            "name": "missing_source_pipeline",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
            # Missing input_source
        }
        
        result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "input_source" in result["error"].lower()


class TestGetWorkflowStatusTool:
    """Test get_workflow_status_tool function."""
    
    @pytest.fixture
    async def running_workflow(self, temp_dir):
        """Create a running workflow for testing."""
        workflow_config = {
            "name": "test_status_workflow",
            "execution_mode": "async",
            "steps": [
                {
                    "name": "slow_step",
                    "type": "processor",
                    "parameters": {"delay": 0.5}
                }
            ]
        }
        
        result = await execute_workflow_tool(
            workflow_config=workflow_config,
            async_execution=True
        )
        
        return result["workflow_id"], result["execution_id"]
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_running(self, running_workflow):
        """Test getting status of a running workflow."""
        workflow_id, execution_id = running_workflow
        
        result = await get_workflow_status_tool(
            workflow_id=workflow_id,
            execution_id=execution_id
        )
        
        assert result["success"] is True
        assert result["status"] in ["running", "completed", "failed"]
        assert "workflow_id" in result
        assert "execution_id" in result
        assert "start_time" in result
        assert "current_step" in result
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_detailed(self, running_workflow):
        """Test getting detailed workflow status."""
        workflow_id, execution_id = running_workflow
        
        result = await get_workflow_status_tool(
            workflow_id=workflow_id,
            execution_id=execution_id,
            include_step_details=True
        )
        
        assert result["success"] is True
        assert "step_details" in result
        assert "progress_percentage" in result
        if result["status"] == "completed":
            assert "end_time" in result
            assert "total_duration" in result
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_with_logs(self, running_workflow):
        """Test getting workflow status with execution logs."""
        workflow_id, execution_id = running_workflow
        
        result = await get_workflow_status_tool(
            workflow_id=workflow_id,
            execution_id=execution_id,
            include_logs=True,
            log_limit=100
        )
        
        assert result["success"] is True
        assert "logs" in result
        assert isinstance(result["logs"], list)
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_nonexistent(self):
        """Test getting status of non-existent workflow."""
        result = await get_workflow_status_tool(
            workflow_id="nonexistent_workflow",
            execution_id="nonexistent_execution"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_by_name(self):
        """Test getting workflow status by workflow name."""
        # First create a workflow
        workflow_config = {
            "name": "named_test_workflow",
            "steps": [{"name": "test_step", "type": "processor"}]
        }
        
        create_result = await execute_workflow_tool(workflow_config=workflow_config)
        
        # Get status by name
        result = await get_workflow_status_tool(workflow_name="named_test_workflow")
        
        assert result["success"] is True
        assert "executions" in result
        assert len(result["executions"]) >= 1


class TestListWorkflowsTool:
    """Test list_workflows_tool function."""
    
    @pytest.fixture
    async def multiple_workflows(self, temp_dir):
        """Create multiple workflows for testing."""
        workflows = []
        
        for i in range(3):
            workflow_config = {
                "name": f"test_workflow_{i}",
                "category": "test",
                "steps": [
                    {
                        "name": f"step_{i}",
                        "type": "processor",
                        "parameters": {"index": i}
                    }
                ]
            }
            
            result = await execute_workflow_tool(workflow_config=workflow_config)
            workflows.append((result["workflow_id"], result["execution_id"]))
        
        return workflows
    
    @pytest.mark.asyncio
    async def test_list_all_workflows(self, multiple_workflows):
        """Test listing all workflows."""
        result = await list_workflows_tool()
        
        assert result["success"] is True
        assert "workflows" in result
        assert "total_count" in result
        assert len(result["workflows"]) >= 3  # At least our test workflows
        
        # Check workflow structure
        for workflow in result["workflows"]:
            assert "workflow_id" in workflow
            assert "name" in workflow
            assert "status" in workflow
            assert "created_at" in workflow
    
    @pytest.mark.asyncio
    async def test_list_workflows_filtered_by_status(self, multiple_workflows):
        """Test listing workflows filtered by status."""
        result = await list_workflows_tool(
            filter_status="completed",
            limit=10
        )
        
        assert result["success"] is True
        assert "workflows" in result
        
        # All returned workflows should have the specified status
        for workflow in result["workflows"]:
            assert workflow["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_list_workflows_filtered_by_category(self, multiple_workflows):
        """Test listing workflows filtered by category."""
        result = await list_workflows_tool(
            filter_category="test",
            limit=5
        )
        
        assert result["success"] is True
        assert "workflows" in result
        
        # All returned workflows should have the specified category
        for workflow in result["workflows"]:
            assert workflow.get("category") == "test"
    
    @pytest.mark.asyncio
    async def test_list_workflows_with_pagination(self, multiple_workflows):
        """Test listing workflows with pagination."""
        # Get first page
        first_page = await list_workflows_tool(limit=2, offset=0)
        
        assert first_page["success"] is True
        assert len(first_page["workflows"]) <= 2
        
        # Get second page
        second_page = await list_workflows_tool(limit=2, offset=2)
        
        assert second_page["success"] is True
        
        # Workflows should be different (if there are enough workflows)
        if len(first_page["workflows"]) == 2 and len(second_page["workflows"]) > 0:
            first_ids = {w["workflow_id"] for w in first_page["workflows"]}
            second_ids = {w["workflow_id"] for w in second_page["workflows"]}
            assert first_ids.isdisjoint(second_ids)
    
    @pytest.mark.asyncio
    async def test_list_workflows_sorted(self, multiple_workflows):
        """Test listing workflows with sorting."""
        # Sort by creation time (newest first)
        result = await list_workflows_tool(
            sort_by="created_at",
            sort_order="desc",
            limit=10
        )
        
        assert result["success"] is True
        assert "workflows" in result
        
        # Check sorting
        if len(result["workflows"]) > 1:
            for i in range(len(result["workflows"]) - 1):
                current_time = result["workflows"][i]["created_at"]
                next_time = result["workflows"][i + 1]["created_at"]
                assert current_time >= next_time
    
    @pytest.mark.asyncio
    async def test_list_workflows_search(self):
        """Test searching workflows by name."""
        # Create a workflow with specific name
        workflow_config = {
            "name": "searchable_unique_workflow",
            "steps": [{"name": "test", "type": "processor"}]
        }
        
        await execute_workflow_tool(workflow_config=workflow_config)
        
        # Search for it
        result = await list_workflows_tool(search_term="searchable_unique")
        
        assert result["success"] is True
        assert "workflows" in result
        
        # Should find the workflow
        found = any(w["name"] == "searchable_unique_workflow" for w in result["workflows"])
        assert found
    
    @pytest.mark.asyncio
    async def test_list_workflows_with_details(self, multiple_workflows):
        """Test listing workflows with detailed information."""
        result = await list_workflows_tool(
            include_execution_details=True,
            include_step_counts=True,
            limit=5
        )
        
        assert result["success"] is True
        assert "workflows" in result
        
        for workflow in result["workflows"]:
            assert "execution_details" in workflow
            assert "step_count" in workflow
            if "executions" in workflow["execution_details"]:
                assert isinstance(workflow["execution_details"]["executions"], list)


class TestWorkflowToolsIntegration:
    """Integration tests for workflow orchestration tools."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self, temp_dir):
        """Test complete workflow lifecycle from creation to completion."""
        # 1. Create embedding pipeline
        pipeline_config = {
            "name": "integration_test_pipeline",
            "input_source": "test_data.jsonl",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 10
        }
        
        pipeline_result = await create_embedding_pipeline_tool(
            pipeline_config=pipeline_config,
            workspace=temp_dir
        )
        assert pipeline_result["success"] is True
        
        # 2. Execute the workflow
        execute_result = await execute_workflow_tool(
            workflow_config=pipeline_result["workflow_config"],
            async_execution=False  # Synchronous for testing
        )
        assert execute_result["success"] is True
        workflow_id = execute_result["workflow_id"]
        execution_id = execute_result["execution_id"]
        
        # 3. Check status
        status_result = await get_workflow_status_tool(
            workflow_id=workflow_id,
            execution_id=execution_id,
            include_step_details=True
        )
        assert status_result["success"] is True
        assert status_result["status"] in ["completed", "running"]
        
        # 4. List workflows to verify it's there
        list_result = await list_workflows_tool(search_term="integration_test")
        assert list_result["success"] is True
        found = any(w["workflow_id"] == workflow_id for w in list_result["workflows"])
        assert found
    
    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, temp_dir):
        """Test parallel execution of multiple workflows."""
        workflows = []
        
        # Create multiple workflow configs
        for i in range(3):
            pipeline_config = {
                "name": f"parallel_pipeline_{i}",
                "input_source": f"data_{i}.jsonl",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
            pipeline_result = await create_embedding_pipeline_tool(
                pipeline_config=pipeline_config,
                workspace=temp_dir
            )
            workflows.append(pipeline_result["workflow_config"])
        
        # Execute workflows in parallel
        execute_tasks = [
            execute_workflow_tool(workflow_config=config, async_execution=True)
            for config in workflows
        ]
        
        results = await asyncio.gather(*execute_tasks, return_exceptions=True)
        
        # Check that all workflows started successfully
        success_count = sum(
            1 for r in results 
            if isinstance(r, dict) and r.get("success") is True
        )
        assert success_count == 3
        
        # Check workflow status
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                status = await get_workflow_status_tool(
                    workflow_id=result["workflow_id"],
                    execution_id=result["execution_id"]
                )
                assert status["success"] is True
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, temp_dir):
        """Test workflow error handling and recovery."""
        # Create a workflow that should fail
        failing_workflow = {
            "name": "failing_workflow",
            "steps": [
                {
                    "name": "invalid_step",
                    "type": "nonexistent_processor",
                    "parameters": {"invalid": "params"}
                }
            ]
        }
        
        # Execute the failing workflow
        result = await execute_workflow_tool(
            workflow_config=failing_workflow,
            continue_on_error=False
        )
        
        # Should handle the error gracefully
        if result["success"]:
            # If execution started, check that it eventually fails
            status = await get_workflow_status_tool(
                workflow_id=result["workflow_id"],
                execution_id=result["execution_id"],
                include_logs=True
            )
            
            assert status["success"] is True
            # Status should eventually be "failed" or contain error information
            assert status["status"] == "failed" or "error" in status
        else:
            # Should fail with meaningful error
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_workflow_monitoring_and_metrics(self, temp_dir):
        """Test workflow monitoring and metrics collection."""
        # Create and execute a workflow
        workflow_config = {
            "name": "monitored_workflow",
            "monitoring": {"enabled": True, "collect_metrics": True},
            "steps": [
                {
                    "name": "monitored_step",
                    "type": "processor",
                    "parameters": {"duration": 0.1}
                }
            ]
        }
        
        execute_result = await execute_workflow_tool(
            workflow_config=workflow_config
        )
        assert execute_result["success"] is True
        
        # Get detailed status with metrics
        status_result = await get_workflow_status_tool(
            workflow_id=execute_result["workflow_id"],
            execution_id=execute_result["execution_id"],
            include_step_details=True,
            include_logs=True
        )
        
        assert status_result["success"] is True
        
        # Should include timing and performance metrics
        if "step_details" in status_result:
            for step in status_result["step_details"]:
                assert "start_time" in step
                if step["status"] == "completed":
                    assert "end_time" in step
                    assert "duration" in step
