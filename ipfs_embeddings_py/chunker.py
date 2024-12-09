import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union
from schema import Document
from llama_index.core.schema import Document
from transformers import AutoTokenizer
from huggingface import HuggingFaceEmbedding
from node_parser import SemanticSplitterNodeParser
import pysbd
import torch
# from node_parser import *
# import llama_index
# from llama_index.core.node_parser import SemanticSplitterNodeParser
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

CHUNKING_STRATEGIES = ['semantic', 'fixed', 'sentences', 'sliding_window']

class chunker:
    def __init__(self, resources=None, metadata=None):
        if resources is None:
            resources = {}
        if metadata is None:  
            metadata = {}
        self.resources = resources
        self.metadata = metadata
        if "chunking_strategy" in metadata.keys():
            chunking_strategy = metadata["chunking_strategy"]
        else:
            chunking_strategy = "semantic"
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        if "models" in metadata.keys():
            if len(list(metadata["models"])) > 0:
                self.embedding_model_name = metadata["models"][0]
                self.embed_model = metadata["models"][0]
            else:
                self.embedding_model_name = None
                self.embed_model = None
        else:
            self.embedding_model_name = None
            self.embed_model = None
        self.chunkers = {}
        if self.embedding_model_name is not None:
            self.chunkers[self.embedding_model_name] = {}
        # self.chunker = self._setup_semantic_chunking(self.embedding_model_name)
        # self.chunkers[self.embedding_model_name]["cpu"] = self.chunker
        self.batch_size = 1
        self.device = None


    def _setup_semantic_chunking(self, embedding_model_name, device=None, target_devices=None, embed_batch_size=None):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name
        
        if embed_batch_size is not None:
            self.batch_size = embed_batch_size
            
        if device is not None:
            if self.device is None:
                self.device = 'cpu'
            self.device = device
            
        if embed_batch_size is None:
            embed_batch_size = 1
        
        if "chunkers" not in self.__dict__.keys():
            self.chunkers = {}
        
        if embedding_model_name not in self.chunkers.keys():
            self.chunkers[embedding_model_name] = {}
        
        if device not in list(self.chunkers[embedding_model_name].keys()): 
                        
            self.chunkers[embedding_model_name][device] = SemanticSplitterNodeParser(
                embed_model=HuggingFaceEmbedding(
                    model_name=self.embedding_model_name,
                    trust_remote_code=True,
                    embed_batch_size=(int(64)),
                    # parallel_process=True,
                    device=device,
                    target_devices=target_devices,
                ),
                show_progress=False,
            )
        
        return None
    
    async def delete_endpoint(self, model_name, endpoint):
        del self.chunkers[model_name][endpoint]
        with torch.no_grad():
            torch.cuda.empty_cache()
        return None

    def chunk_semantically(
        self,
        text: str,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, device='cpu', use_fast=True)
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        
        if embedding_model_name is None:
            self._setup_semantic_chunking(self.embedding_model_name, device, None, batch_size)
        else:
            self._setup_semantic_chunking(embedding_model_name, device, None, batch_size)

        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.chunkers[embedding_model_name][device].get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]
        
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
        device: Optional[str] = None,
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
        device: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        if embedding_model_name is None and self.embedding_model_name is not None:
            embedding_model_name = self.embedding_model_name
        if tokenizer is None and self.embedding_model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, use_fast=True, device='cpu')
        elif tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if n_sentences is None:
            n_sentences = 8
        seg = pysbd.Segmenter(language="en", clean=False)
        segments = seg.segment(text)
        chunk_spans = []
        count_chunks = 0
        count_sentences = 0
        chunk_start = 0
        count_tokens = 0
        for i, segment in enumerate(segments):
            count_sentences += 1
            tokens = tokenizer.encode_plus(
                segment, return_offsets_mapping=True, add_special_tokens=False
            )
            count_tokens += len(tokens.offset_mapping)
            if count_sentences == n_sentences:
                chunk_spans.append((chunk_start, count_tokens + 1))
                chunk_start = count_tokens + 1
                count_chunks += 1
                count_sentences = 0
        if count_tokens - chunk_start > 1:
            chunk_spans.append((chunk_start, count_tokens))
        return chunk_spans
    
    def chunk_by_sliding_window(
        self,
        text: str,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        tokenizer: Optional['AutoTokenizer'] = None,
        embedding_model_name: Optional[str] = None,
        device: Optional[str] = None,
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
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
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
            return self.chunk_semantically(text, tokenizer, embedding_model_name, device, batch_size)
        elif chunking_strategy == "fixed":
            if chunk_size < 4:
                chunk_size = 4
            return self.chunk_by_tokens(text, chunk_size, tokenizer, embedding_model_name, device)
        elif chunking_strategy == "sentences":
            return self.chunk_by_sentences(text, n_sentences, tokenizer, embedding_model_name, device)
        elif chunking_strategy == "sliding_window":
            return self.chunk_by_sliding_window(text, chunk_size, step_size, tokenizer, embedding_model_name, device)
        else:
            raise ValueError("Unsupported chunking strategy")
        
__all__ = ['chunker']