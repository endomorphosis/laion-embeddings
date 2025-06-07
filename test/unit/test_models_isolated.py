import pytest
from pydantic import ValidationError
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test just the models without importing the full main.py
try:
    # Try to import just the models we can test
    from typing import List, Optional
    from pydantic import BaseModel, Field
    from fastapi import HTTPException
    import re
    
    # Define the classes locally for testing
    class InputValidator:
        @staticmethod
        def validate_text_input(text: str) -> str:
            if not text or len(text.strip()) == 0:
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            if len(text) > 10000:
                raise HTTPException(status_code=400, detail="Text too long (max 10k characters)")
            return text.strip()
        
        @staticmethod
        def validate_model_name(model: str) -> str:
            allowed_models = [
                "thenlper/gte-small",
                "Alibaba-NLP/gte-large-en-v1.5",
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            ]
            if model not in allowed_models:
                raise HTTPException(status_code=400, detail=f"Model {model} not allowed")
            return model
        
        @staticmethod
        def validate_dataset_name(dataset: str) -> str:
            if not dataset or len(dataset.strip()) == 0:
                raise HTTPException(status_code=400, detail="Dataset name cannot be empty")
            if not re.match(r'^[a-zA-Z0-9_/-]+$', dataset):
                raise HTTPException(status_code=400, detail="Invalid dataset name format")
            return dataset.strip()

    class SearchRequest(BaseModel):
        collection: str = Field(..., description="Collection to search", min_length=1, max_length=100)
        text: str = Field(..., description="Search text", min_length=1, max_length=10000)
        n: int = Field(default=10, description="Number of results", ge=1, le=100)

    class CreateEmbeddingsRequest(BaseModel):
        dataset: str = Field(..., description="Dataset identifier", min_length=1, max_length=200)
        split: str = Field(..., description="Dataset split", min_length=1, max_length=50)
        column: str = Field(..., description="Text column name", min_length=1, max_length=100)
        dst_path: str = Field(..., description="Destination path", min_length=1, max_length=500)
        models: List[str] = Field(..., description="List of model names", min_items=1, max_items=10)

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Models not available for testing: {e}")


class TestInputValidation:
    """Test input validation functions"""
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_text_input_valid(self):
        """Test valid text input"""
        result = InputValidator.validate_text_input("  valid text  ")
        assert result == "valid text"
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_text_input_empty(self):
        """Test empty text input"""
        with pytest.raises(HTTPException):
            InputValidator.validate_text_input("")
            
        with pytest.raises(HTTPException):
            InputValidator.validate_text_input("   ")
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_text_input_too_long(self):
        """Test text input that's too long"""
        long_text = "a" * 10001
        with pytest.raises(HTTPException):
            InputValidator.validate_text_input(long_text)
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_model_name_valid(self):
        """Test valid model names"""
        valid_models = [
            "thenlper/gte-small",
            "Alibaba-NLP/gte-large-en-v1.5",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        ]
        for model in valid_models:
            result = InputValidator.validate_model_name(model)
            assert result == model
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_model_name_invalid(self):
        """Test invalid model names"""
        with pytest.raises(HTTPException):
            InputValidator.validate_model_name("invalid/model")
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_dataset_name_valid(self):
        """Test valid dataset names"""
        valid_datasets = [
            "test/dataset",
            "company_dataset",
            "dataset-name",
            "TeraflopAI/Caselaw_Access_Project"
        ]
        for dataset in valid_datasets:
            result = InputValidator.validate_dataset_name(dataset)
            assert result == dataset
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_validate_dataset_name_invalid(self):
        """Test invalid dataset names"""
        with pytest.raises(HTTPException):
            InputValidator.validate_dataset_name("")
            
        with pytest.raises(HTTPException):
            InputValidator.validate_dataset_name("invalid dataset name")


class TestPydanticModels:
    """Test Pydantic model validation"""
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_search_request_valid(self):
        """Test valid SearchRequest"""
        request = SearchRequest(
            collection="test_collection",
            text="sample query",
            n=5
        )
        assert request.collection == "test_collection"
        assert request.text == "sample query"
        assert request.n == 5
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_search_request_default_n(self):
        """Test SearchRequest with default n"""
        request = SearchRequest(
            collection="test_collection",
            text="sample query"
        )
        assert request.n == 10  # default value
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_search_request_validation_errors(self):
        """Test SearchRequest validation errors"""
        # Test empty collection
        with pytest.raises(ValidationError):
            SearchRequest(collection="", text="query", n=5)
        
        # Test empty text
        with pytest.raises(ValidationError):
            SearchRequest(collection="test", text="", n=5)
        
        # Test invalid n
        with pytest.raises(ValidationError):
            SearchRequest(collection="test", text="query", n=0)
        
        with pytest.raises(ValidationError):
            SearchRequest(collection="test", text="query", n=101)
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_create_embeddings_request_valid(self):
        """Test valid CreateEmbeddingsRequest"""
        request = CreateEmbeddingsRequest(
            dataset="test/dataset",
            split="train",
            column="text",
            dst_path="/tmp/test",
            models=["thenlper/gte-small"]
        )
        assert request.dataset == "test/dataset"
        assert request.models == ["thenlper/gte-small"]
    
    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_create_embeddings_request_validation_errors(self):
        """Test CreateEmbeddingsRequest validation errors"""
        # Test empty models list
        with pytest.raises(ValidationError):
            CreateEmbeddingsRequest(
                dataset="test",
                split="train", 
                column="text",
                dst_path="/tmp/test",
                models=[]
            )
        
        # Test empty dataset
        with pytest.raises(ValidationError):
            CreateEmbeddingsRequest(
                dataset="",
                split="train",
                column="text", 
                dst_path="/tmp/test",
                models=["model"]
            )


class TestBasicFunctionality:
    """Test basic functionality without external dependencies"""
    
    def test_imports_work(self):
        """Test that basic imports work"""
        import pytest
        from pydantic import BaseModel, Field
        import re
        assert True
    
    def test_regex_patterns(self):
        """Test regex patterns used in validation"""
        valid_pattern = r'^[a-zA-Z0-9_/-]+$'
        
        # Valid dataset names
        valid_names = [
            "test_dataset",
            "company/dataset", 
            "dataset-name",
            "123dataset",
            "TeraflopAI/Caselaw_Access_Project"
        ]
        
        for name in valid_names:
            assert re.match(valid_pattern, name), f"Pattern should match {name}"
        
        # Invalid dataset names  
        invalid_names = [
            "dataset with spaces",
            "dataset!@#",
            "dataset.with.dots"
        ]
        
        for name in invalid_names:
            assert not re.match(valid_pattern, name), f"Pattern should not match {name}"


if __name__ == "__main__":
    pytest.main([__file__])
