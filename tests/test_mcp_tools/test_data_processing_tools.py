# tests/test_mcp_tools/test_data_processing_tools.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np

from src.mcp_server.tools.data_processing_tools import (
    ChunkingTool,
    # DataTransformationTool,
    # DataValidationTool,
    # DataConversionTool
)
from src.mcp_server.error_handlers import MCPError, ValidationError


class TestChunkingTool:
    """Test cases for ChunkingTool."""

    @pytest.fixture
    def mock_ipfs_embeddings(self):
        """Mock IPFS embeddings instance."""
        ipfs_instance = Mock()
        ipfs_instance.chunk_text = AsyncMock(return_value=[
            {"chunk_id": "chunk1", "content": "First chunk of text"},
            {"chunk_id": "chunk2", "content": "Second chunk of text"}
        ])
        return ipfs_instance

    @pytest.fixture
    def mock_chunker(self):
        """Mock chunker instance."""
        chunker = Mock()
        chunker.chunk_by_tokens = Mock(return_value=[
            "First chunk of text content that is longer",
            "Second chunk with different content here"
        ])
        chunker.chunk_by_sentences = Mock(return_value=[
            "First sentence. Second sentence.",
            "Third sentence. Fourth sentence."
        ])
        chunker.chunk_semantically = Mock(return_value=[
            "Semantically related content group one",
            "Semantically related content group two"
        ])
        return chunker

    @pytest.fixture
    def chunking_tool(self, mock_ipfs_embeddings):
        """Create ChunkingTool instance."""
        tool = ChunkingTool(mock_ipfs_embeddings)
        return tool

    @pytest.mark.asyncio
    async def test_token_based_chunking(self, chunking_tool, mock_chunker):
        """Test token-based text chunking."""
        chunking_tool.chunker_instance = mock_chunker
        
        parameters = {
            "text": "This is a long text that needs to be chunked into smaller pieces based on token count.",
            "method": "tokens",
            "chunk_size": 256
        }

        result = await chunking_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["method"] == "tokens"
        assert result["chunk_size"] == 256
        assert "chunks" in result
        assert len(result["chunks"]) == 2
        assert result["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_sentence_based_chunking(self, chunking_tool, mock_chunker):
        """Test sentence-based text chunking."""
        chunking_tool.chunker_instance = mock_chunker
        
        parameters = {
            "text": "First sentence. Second sentence. Third sentence. Fourth sentence.",
            "method": "sentences",
            "n_sentences": 2
        }

        result = await chunking_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["method"] == "sentences"
        assert result["n_sentences"] == 2
        assert "chunks" in result
        assert len(result["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_semantic_chunking(self, chunking_tool, mock_chunker):
        """Test semantic text chunking."""
        chunking_tool.chunker_instance = mock_chunker
        
        parameters = {
            "text": "Content about topic A. More content about topic A. Now content about topic B. Additional topic B content.",
            "method": "semantic",
            "similarity_threshold": 0.7
        }

        result = await chunking_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["method"] == "semantic"
        assert "chunks" in result
        assert len(result["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_sliding_window_chunking(self, chunking_tool, mock_chunker):
        """Test sliding window text chunking."""
        chunking_tool.chunker_instance = mock_chunker
        mock_chunker.chunk_sliding_window = Mock(return_value=[
            "First sliding window chunk with overlap",
            "Second sliding window chunk with overlap"
        ])
        
        parameters = {
            "text": "Long text content for sliding window chunking with overlapping segments",
            "method": "sliding_window",
            "chunk_size": 128,
            "overlap": 32
        }

        result = await chunking_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["method"] == "sliding_window"
        assert result["chunk_size"] == 128
        assert result["overlap"] == 32

    @pytest.mark.asyncio
    async def test_empty_text_input(self, chunking_tool):
        """Test handling of empty text input."""
        parameters = {
            "text": "",
            "method": "tokens"
        }

        with pytest.raises(ValidationError, match="Text input cannot be empty"):
            await chunking_tool.execute(parameters)

    @pytest.mark.asyncio
    async def test_invalid_chunking_method(self, chunking_tool):
        """Test handling of invalid chunking method."""
        parameters = {
            "text": "Some text to chunk",
            "method": "invalid_method"
        }

        with pytest.raises(ValidationError):
            await chunking_tool.execute(parameters)

    @pytest.mark.parametrize("method,expected_params", [
        ("tokens", {"chunk_size": 256}),
        ("sentences", {"n_sentences": 3}),
        ("semantic", {"similarity_threshold": 0.7}),
        ("sliding_window", {"chunk_size": 128, "overlap": 32})
    ])
    @pytest.mark.asyncio
    async def test_method_specific_parameters(self, chunking_tool, mock_chunker, method, expected_params):
        """Test method-specific parameter handling."""
        chunking_tool.chunker_instance = mock_chunker
        
        parameters = {
            "text": "Sample text for chunking",
            "method": method,
            **expected_params
        }

        result = await chunking_tool.execute(parameters)

        assert result["status"] == "success"
        assert result["method"] == method


# class TestDataTransformationTool:
#     """Test cases for DataTransformationTool."""

#     @pytest.fixture
#     def mock_data_service(self):
#         """Mock data service."""
#         service = Mock()
#         service.transform_data = AsyncMock(return_value={
#             "transformed_data": [{"id": 1, "text": "TRANSFORMED TEXT"}],
#             "transformation_count": 1
#         })
#         return service

#     @pytest.fixture
#     def transformation_tool(self, mock_data_service):
#         """Create DataTransformationTool instance."""
#         return DataTransformationTool(data_service=mock_data_service)

#     @pytest.mark.asyncio
#     async def test_text_normalization(self, transformation_tool):
#         """Test text normalization transformation."""
#         parameters = {
#             "data": [{"id": 1, "text": "  MIXED case Text  "}],
#             "transformation_type": "normalize_text",
#             "options": {
#                 "lowercase": True,
#                 "strip_whitespace": True,
#                 "remove_special_chars": False
#             }
#         }

#         result = await transformation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["transformation_type"] == "normalize_text"
#         assert "transformed_data" in result
#         assert result["records_processed"] == 1

#     @pytest.mark.asyncio
#     async def test_data_filtering(self, transformation_tool):
#         """Test data filtering transformation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "text": "Short"},
#                 {"id": 2, "text": "This is a longer text that meets criteria"}
#             ],
#             "transformation_type": "filter",
#             "filter_criteria": {
#                 "field": "text",
#                 "condition": "min_length",
#                 "value": 10
#             }
#         }

#         transformation_tool.data_service.transform_data.return_value = {
#             "transformed_data": [{"id": 2, "text": "This is a longer text that meets criteria"}],
#             "transformation_count": 1,
#             "filtered_count": 1
#         }

#         result = await transformation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["transformation_type"] == "filter"
#         assert result["records_processed"] == 2
#         assert result["records_filtered"] == 1

#     @pytest.mark.asyncio
#     async def test_data_enrichment(self, transformation_tool):
#         """Test data enrichment transformation."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "transformation_type": "enrich",
#             "enrichment_fields": ["word_count", "char_count", "language"]
#         }

#         transformation_tool.data_service.transform_data.return_value = {
#             "transformed_data": [{
#                 "id": 1,
#                 "text": "Sample text",
#                 "word_count": 2,
#                 "char_count": 11,
#                 "language": "en"
#             }],
#             "transformation_count": 1
#         }

#         result = await transformation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["transformation_type"] == "enrich"
#         assert "enrichment_fields" in result

#     @pytest.mark.asyncio
#     async def test_data_deduplication(self, transformation_tool):
#         """Test data deduplication transformation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "text": "Duplicate text"},
#                 {"id": 2, "text": "Duplicate text"},
#                 {"id": 3, "text": "Unique text"}
#             ],
#             "transformation_type": "deduplicate",
#             "dedup_field": "text"
#         }

#         transformation_tool.data_service.transform_data.return_value = {
#             "transformed_data": [
#                 {"id": 1, "text": "Duplicate text"},
#                 {"id": 3, "text": "Unique text"}
#             ],
#             "transformation_count": 2,
#             "duplicates_removed": 1
#         }

#         result = await transformation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["transformation_type"] == "deduplicate"
#         assert result["duplicates_removed"] == 1

#     @pytest.mark.asyncio
#     async def test_batch_transformations(self, transformation_tool):
#         """Test batch transformations."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "transformation_type": "batch",
#             "transformations": [
#                 {"type": "normalize_text", "options": {"lowercase": True}},
#                 {"type": "enrich", "fields": ["word_count"]},
#                 {"type": "filter", "criteria": {"field": "word_count", "condition": "min", "value": 1}}
#             ]
#         }

#         result = await transformation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["transformation_type"] == "batch"
#         assert "transformations_applied" in result

#     @pytest.mark.asyncio
#     async def test_invalid_transformation_type(self, transformation_tool):
#         """Test handling of invalid transformation type."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "transformation_type": "invalid_type"
#         }

#         with pytest.raises(ValidationError):
#             await transformation_tool.execute(parameters)

#     @pytest.mark.asyncio
#     async def test_empty_data_input(self, transformation_tool):
#         """Test handling of empty data input."""
#         parameters = {
#             "data": [],
#             "transformation_type": "normalize_text"
#         }

#         with pytest.raises(ValidationError, match="Data input cannot be empty"):
#             await transformation_tool.execute(parameters)


# class TestDataValidationTool:
#     """Test cases for DataValidationTool."""

#     @pytest.fixture
#     def mock_validator_service(self):
#         """Mock validator service."""
#         service = Mock()
#         service.validate_data = AsyncMock(return_value={
#             "is_valid": True,
#             "validation_errors": [],
#             "validation_warnings": [],
#             "records_validated": 2
#         })
#         return service

#     @pytest.fixture
#     def validation_tool(self, mock_validator_service):
#         """Create DataValidationTool instance."""
#         return DataValidationTool(validator_service=mock_validator_service)

#     @pytest.mark.asyncio
#     async def test_schema_validation(self, validation_tool):
#         """Test schema validation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "text": "Valid text", "score": 0.95},
#                 {"id": 2, "text": "Another valid text", "score": 0.87}
#             ],
#             "validation_type": "schema",
#             "schema": {
#                 "id": {"type": "integer", "required": True},
#                 "text": {"type": "string", "required": True, "min_length": 1},
#                 "score": {"type": "float", "min": 0.0, "max": 1.0}
#             }
#         }

#         result = await validation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["validation_type"] == "schema"
#         assert result["is_valid"] is True
#         assert result["records_validated"] == 2
#         assert len(result["validation_errors"]) == 0

#     @pytest.mark.asyncio
#     async def test_data_quality_validation(self, validation_tool):
#         """Test data quality validation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "text": "Good quality text with sufficient length"},
#                 {"id": 2, "text": ""}
#             ],
#             "validation_type": "quality",
#             "quality_checks": ["completeness", "uniqueness", "consistency"]
#         }

#         validation_tool.validator_service.validate_data.return_value = {
#             "is_valid": False,
#             "validation_errors": [
#                 {"record_id": 2, "field": "text", "error": "Empty text field"}
#             ],
#             "validation_warnings": [],
#             "records_validated": 2,
#             "quality_score": 0.75
#         }

#         result = await validation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["validation_type"] == "quality"
#         assert result["is_valid"] is False
#         assert len(result["validation_errors"]) == 1
#         assert result["quality_score"] == 0.75

#     @pytest.mark.asyncio
#     async def test_format_validation(self, validation_tool):
#         """Test format validation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "email": "valid@example.com", "url": "https://example.com"},
#                 {"id": 2, "email": "invalid-email", "url": "not-a-url"}
#             ],
#             "validation_type": "format",
#             "format_rules": {
#                 "email": "email",
#                 "url": "url"
#             }
#         }

#         validation_tool.validator_service.validate_data.return_value = {
#             "is_valid": False,
#             "validation_errors": [
#                 {"record_id": 2, "field": "email", "error": "Invalid email format"},
#                 {"record_id": 2, "field": "url", "error": "Invalid URL format"}
#             ],
#             "validation_warnings": [],
#             "records_validated": 2
#         }

#         result = await validation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["validation_type"] == "format"
#         assert len(result["validation_errors"]) == 2

#     @pytest.mark.asyncio
#     async def test_business_rules_validation(self, validation_tool):
#         """Test business rules validation."""
#         parameters = {
#             "data": [
#                 {"id": 1, "price": 100, "discount": 10},
#                 {"id": 2, "price": 50, "discount": 60}
#             ],
#             "validation_type": "business_rules",
#             "rules": [
#                 {"rule": "discount_less_than_price", "description": "Discount must be less than price"}
#             ]
#         }

#         validation_tool.validator_service.validate_data.return_value = {
#             "is_valid": False,
#             "validation_errors": [
#                 {"record_id": 2, "rule": "discount_less_than_price", "error": "Discount exceeds price"}
#             ],
#             "validation_warnings": [],
#             "records_validated": 2
#         }

#         result = await validation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["validation_type"] == "business_rules"
#         assert len(result["validation_errors"]) == 1

#     @pytest.mark.asyncio
#     async def test_custom_validation(self, validation_tool):
#         """Test custom validation rules."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "validation_type": "custom",
#             "custom_validator": "lambda x: len(x.get('text', '')) > 5",
#             "error_message": "Text must be longer than 5 characters"
#         }

#         result = await validation_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["validation_type"] == "custom"

#     @pytest.mark.asyncio
#     async def test_invalid_validation_type(self, validation_tool):
#         """Test handling of invalid validation type."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "validation_type": "invalid_type"
#         }

#         with pytest.raises(ValidationError):
#             await validation_tool.execute(parameters)


# class TestDataConversionTool:
#     """Test cases for DataConversionTool."""

#     @pytest.fixture
#     def mock_converter_service(self):
#         """Mock converter service."""
#         service = Mock()
#         service.convert_format = AsyncMock(return_value={
#             "converted_data": "converted_content",
#             "source_format": "json",
#             "target_format": "parquet",
#             "records_converted": 100
#         })
#         return service

#     @pytest.fixture
#     def conversion_tool(self, mock_converter_service):
#         """Create DataConversionTool instance."""
#         return DataConversionTool(converter_service=mock_converter_service)

#     @pytest.mark.asyncio
#     async def test_json_to_parquet_conversion(self, conversion_tool):
#         """Test JSON to Parquet conversion."""
#         parameters = {
#             "data": [{"id": 1, "text": "Sample text"}],
#             "source_format": "json",
#             "target_format": "parquet",
#             "output_path": "/tmp/output.parquet"
#         }

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["source_format"] == "json"
#         assert result["target_format"] == "parquet"
#         assert result["records_converted"] == 100

#     @pytest.mark.asyncio
#     async def test_csv_to_json_conversion(self, conversion_tool):
#         """Test CSV to JSON conversion."""
#         parameters = {
#             "data": "id,text\n1,Sample text\n2,Another text",
#             "source_format": "csv",
#             "target_format": "json",
#             "conversion_options": {
#                 "delimiter": ",",
#                 "header": True
#             }
#         }

#         conversion_tool.converter_service.convert_format.return_value = {
#             "converted_data": [
#                 {"id": "1", "text": "Sample text"},
#                 {"id": "2", "text": "Another text"}
#             ],
#             "source_format": "csv",
#             "target_format": "json",
#             "records_converted": 2
#         }

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["records_converted"] == 2

#     @pytest.mark.asyncio
#     async def test_format_auto_detection(self, conversion_tool):
#         """Test automatic format detection."""
#         parameters = {
#             "data": '{"id": 1, "text": "Sample text"}',
#             "target_format": "parquet",
#             "auto_detect_format": True
#         }

#         conversion_tool.converter_service.detect_format = AsyncMock(return_value="json")

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "success"
#         conversion_tool.converter_service.detect_format.assert_called_once()

#     @pytest.mark.asyncio
#     async def test_batch_conversion(self, conversion_tool):
#         """Test batch conversion of multiple files."""
#         parameters = {
#             "conversion_type": "batch",
#             "input_files": [
#                 {"path": "/data/file1.json", "format": "json"},
#                 {"path": "/data/file2.csv", "format": "csv"}
#             ],
#             "target_format": "parquet",
#             "output_directory": "/output/"
#         }

#         conversion_tool.converter_service.convert_batch = AsyncMock(return_value={
#             "conversions": [
#                 {"file": "file1.json", "status": "success", "records": 50},
#                 {"file": "file2.csv", "status": "success", "records": 75}
#             ],
#             "total_records": 125,
#             "successful_conversions": 2,
#             "failed_conversions": 0
#         })

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["total_records"] == 125
#         assert result["successful_conversions"] == 2

#     @pytest.mark.asyncio
#     async def test_unsupported_format_conversion(self, conversion_tool):
#         """Test handling of unsupported format conversion."""
#         conversion_tool.converter_service.convert_format.side_effect = Exception("Unsupported format conversion")

#         parameters = {
#             "data": "some data",
#             "source_format": "unsupported",
#             "target_format": "json"
#         }

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "error"
#         assert "Unsupported format conversion" in result["error"]

#     @pytest.mark.parametrize("source_format,target_format", [
#         ("json", "parquet"),
#         ("csv", "json"),
#         ("parquet", "csv"),
#         ("xml", "json"),
#         ("yaml", "json")
#     ])
#     @pytest.mark.asyncio
#     async def test_format_combinations(self, conversion_tool, source_format, target_format):
#         """Test various format conversion combinations."""
#         parameters = {
#             "data": "sample_data",
#             "source_format": source_format,
#             "target_format": target_format
#         }

#         result = await conversion_tool.execute(parameters)

#         assert result["status"] == "success"
#         assert result["source_format"] == source_format
#         assert result["target_format"] == target_format
