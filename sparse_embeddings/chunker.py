import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union
import llama_index
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

CHUNKING_STRATEGIES = ['semantic', 'fixed', 'sentences', 'sliding_window']

class Chunker:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        if "chunking_strategy" in metadata.keys():
            chunking_strategy = metadata["chunking_strategy"]
        else:
            chunking_strategy = "semantic"
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        
        if len(list(metadata["models"])) > 0:
            self.embedding_model_name = metadata["models"][0]
            self.embed_model = metadata["models"][0]
        else:
            self.embedding_model_name = None
            self.embed_model = None

    def _setup_semantic_chunking(self, embedding_model_name):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            embed_batch_size=1,
        )
        self.splitter = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            show_progress=False,
        )

    def chunk_semantically(
        self,
        text: str,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, device='cpu', use_fast=True)
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if self.embed_model is not None:
            if embedding_model_name is None:
                self._setup_semantic_chunking(self.embedding_model_name)
            else:
                self._setup_semantic_chunking(embedding_model_name)

        # Get semantic nodes
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]

        # Tokenize the entire text
        tokens = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []

        for char_start, char_end in nodes:
            # Convert char indices to token indices
            start_chunk_index = bisect.bisect_left(
                [offset[0] for offset in token_offsets], char_start
            )
            end_chunk_index = bisect.bisect_right(
                [offset[1] for offset in token_offsets], char_end
            )

            # Add the chunk span if it's within the tokenized text
            if start_chunk_index < len(token_offsets) and end_chunk_index <= len(
                token_offsets
            ):
                chunk_spans.append((start_chunk_index, end_chunk_index))
            else:
                break

        return chunk_spans

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, use_fast=True, device='cpu')
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if chunk_size is None:
            chunk_size = 512
        if chunk_size < 4:
            chunk_size = 4

        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        for i in range(0, len(token_offsets), chunk_size):
            chunk_end = min(i + chunk_size, len(token_offsets))
            if chunk_end - i > 0:
                chunk_spans.append((i, chunk_end))

        return chunk_spans

    def chunk_by_sentences(
        self,
        text: str,
        n_sentences: Optional[int] = None,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, use_fast=True, device='cpu')
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if n_sentences is None:
            n_sentences = 8

        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        chunk_start = 0
        count_chunks = 0
        for i in range(0, len(token_offsets)):
            if tokens.tokens(0)[i] in ('.', '!', '?') and (
                (len(tokens.tokens(0)) == i + 1)
                or (tokens.token_to_chars(i).end != tokens.token_to_chars(i + 1).start)
            ):
                count_chunks += 1
                if count_chunks == n_sentences:
                    chunk_spans.append((chunk_start, i + 1))
                    chunk_start = i + 1
                    count_chunks = 0
        if len(tokens.tokens(0)) - chunk_start > 1:
            chunk_spans.append((chunk_start, len(tokens.tokens(0))))
        return chunk_spans
    
    def chunk_by_sliding_window(
        self,
        text: str,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, use_fast=True, device='cpu')
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if window_size is None:
            window_size = 512
        if step_size is None:
            step_size = 256

        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        for i in range(0, len(token_offsets), step_size):
            chunk_end = min(i + window_size, len(token_offsets))
            if chunk_end - i > 0:
                chunk_spans.append((i, chunk_end))

        return chunk_spans

    def chunk(
        self,
        text: str,
        tokenizer: Optional['AutoTokenizer'] = None,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
        step_size: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
    ):
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, use_fast=True, device='cpu')
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        if chunk_size is None:
            chunk_size = 512
        if n_sentences is None:
            n_sentences = 8
        if step_size is None:
            step_size = 256
        if chunking_strategy is None:
            chunking_strategy = "semantic"
        
        chunking_strategy = chunking_strategy or self.chunking_strategy
        if chunking_strategy == "semantic":
            return self.chunk_semantically(text, tokenizer, embedding_model_name)
        elif chunking_strategy == "fixed":
            if chunk_size < 4:
                chunk_size = 4
            return self.chunk_by_tokens(text, chunk_size, tokenizer, embedding_model_name)
        elif chunking_strategy == "sentences":
            return self.chunk_by_sentences(text, n_sentences, tokenizer, embedding_model_name)
        elif chunking_strategy == "sliding_window":
            return self.chunk_by_sliding_window(text, chunk_size, step_size, tokenizer, embedding_model_name)
        else:
            raise ValueError("Unsupported chunking strategy")