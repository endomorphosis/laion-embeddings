"""
Embedding Service

This module provides embedding generation capabilities using various models
including Sentence Transformers, OpenAI, and other providers.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 batch_size: int = 32):
        """
        Initialize Sentence Transformers provider.
        
        Args:
            model_name: Name of the model to use
            device: Device to use (cpu, cuda, etc.)
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model = None
        self._dimension = None
    
    async def initialize(self) -> None:
        """Initialize the model."""
        try:
            # Import sentence-transformers here to avoid import errors if not installed
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Get dimension by encoding a test sentence
            test_embedding = self.model.encode(["test"], normalize_embeddings=self.normalize_embeddings)
            self._dimension = len(test_embedding[0])
            
            logger.info(f"Initialized SentenceTransformers model: {self.model_name} (dim: {self._dimension})")
            
        except ImportError:
            raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformers model: {e}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        if self.model is None:
            await self.initialize()
        
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True
                )
                all_embeddings.extend(embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._dimension
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.model = None


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "text-embedding-ada-002",
                 batch_size: int = 100):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            batch_size: Batch size for processing
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.client = None
        self._dimension = None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
            
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Get dimension by encoding a test sentence
            test_response = await self.client.embeddings.create(
                input=["test"],
                model=self.model
            )
            self._dimension = len(test_response.data[0].embedding)
            
            logger.info(f"Initialized OpenAI embedding model: {self.model} (dim: {self._dimension})")
            
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        if self.client is None:
            await self.initialize()
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        return self._dimension
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.close()
        self.client = None


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace Transformers embedding provider."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 batch_size: int = 32):
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name: Name of the model to use
            device: Device to use
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device or "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self._dimension = None
    
    async def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get dimension
            with torch.no_grad():
                test_inputs = self.tokenizer(["test"], return_tensors="pt", padding=True, truncation=True)
                test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
                outputs = self.model(**test_inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                self._dimension = embeddings.shape[1]
            
            logger.info(f"Initialized HuggingFace model: {self.model_name} (dim: {self._dimension})")
            
        except ImportError:
            raise ImportError("transformers and torch packages are required. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {e}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        if self.model is None:
            await self.initialize()
        
        try:
            import torch
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    if self.normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate HuggingFace embeddings: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._dimension
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.model = None
        self.tokenizer = None


class EmbeddingService:
    """
    Main embedding service that manages different providers.
    """
    
    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize embedding service.
        
        Args:
            provider: Embedding provider instance
            config: Configuration dictionary
        """
        self.config = config or {}
        self.provider = provider or self._create_default_provider()
        self._is_ready = False
    
    def _create_default_provider(self) -> BaseEmbeddingProvider:
        """Create default embedding provider based on configuration."""
        provider_type = self.config.get('provider', 'sentence-transformers')
        
        if provider_type == 'sentence-transformers':
            return SentenceTransformersProvider(
                model_name=self.config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                normalize_embeddings=self.config.get('normalize', True),
                batch_size=self.config.get('batch_size', 32)
            )
        elif provider_type == 'openai':
            return OpenAIEmbeddingProvider(
                api_key=self.config.get('api_key'),
                model=self.config.get('model', 'text-embedding-ada-002'),
                batch_size=self.config.get('batch_size', 100)
            )
        elif provider_type == 'huggingface':
            return HuggingFaceEmbeddingProvider(
                model_name=self.config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                normalize_embeddings=self.config.get('normalize', True),
                batch_size=self.config.get('batch_size', 32)
            )
        else:
            # Default to sentence-transformers
            return SentenceTransformersProvider()
    
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        if not self._is_ready:
            await self.provider.initialize()
            self._is_ready = True
            logger.info("Embedding service initialized")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        if not self._is_ready:
            await self.initialize()
        
        return await self.provider.embed_documents(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        if not self._is_ready:
            await self.initialize()
        
        return await self.provider.embed_query(text)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._is_ready:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        return self.provider.get_dimension()
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self._is_ready
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.provider:
            await self.provider.cleanup()
        self._is_ready = False
        logger.info("Embedding service cleaned up")


def create_embedding_service(config: Optional[Dict[str, Any]] = None) -> EmbeddingService:
    """
    Create an embedding service with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(config=config)
