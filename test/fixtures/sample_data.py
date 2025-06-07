"""
Sample data fixtures for testing
"""

SAMPLE_EMBEDDINGS = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0],
    [0.2, 0.3, 0.4, 0.5, 0.6]
]

SAMPLE_TEXTS = [
    "This is a sample text for testing embeddings.",
    "Another sample text with different content.",
    "A third example text for comprehensive testing."
]

SAMPLE_DATASETS = {
    "small_test": {
        "dataset": "test/small-dataset",
        "split": "train",
        "column": "text",
        "size": 100
    },
    "medium_test": {
        "dataset": "test/medium-dataset", 
        "split": "validation",
        "column": "content",
        "size": 1000
    }
}

SAMPLE_MODELS = [
    "thenlper/gte-small",
    "Alibaba-NLP/gte-large-en-v1.5"
]

SAMPLE_SEARCH_RESULTS = [
    {
        "id": "doc_1",
        "text": "Sample document text",
        "score": 0.95,
        "metadata": {"source": "test"}
    },
    {
        "id": "doc_2", 
        "text": "Another document with relevant content",
        "score": 0.87,
        "metadata": {"source": "test"}
    }
]

INVALID_INPUTS = {
    "empty_text": "",
    "too_long_text": "a" * 10001,
    "invalid_model": "non-existent/model",
    "invalid_dataset": "dataset with spaces",
    "negative_n": -1,
    "zero_n": 0,
    "too_large_n": 1000
}
