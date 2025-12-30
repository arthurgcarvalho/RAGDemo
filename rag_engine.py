import os
import google.generativeai as genai
import faiss
import numpy as np
from typing import List, Tuple, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGDemo:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # genai.configure(api_key=self.api_key) # Deprecated
        # Using new setup if available or just relying on module level config if that's what user has.
        # But based on prev code, we need to configure it:
        genai.configure(api_key=self.api_key)
        
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.dimension = 768 # Dimensions for gemini-embedding-001

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Uses LangChain's RecursiveCharacterTextSplitter to split text 
        respecting natural boundaries (paragraphs, newlines, spaces).
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50, # Modest overlap to maintain context
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_text(text)

    def get_embedding(self, text: str) -> np.ndarray:
        """Fetch embedding for a single text."""
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Embedding of chunk"
        )
        return np.array(result['embedding'])

    def ingest(self, text: str, chunk_size: int):
        """Processes the text, chunks it, embeds it, and builds the FAISS index."""
        # 1. Chunking
        self.chunks = self.chunk_text(text, chunk_size)
        if not self.chunks:
            return 0

        # 2. Embedding (Batched for performance)
        embeddings_list = []
        batch_size = 100
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            try:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=batch,
                    task_type="retrieval_document",
                    title="Embedding of chunk"
                )
                # If batch has 1 item, API might return single embedding dict, need to handle list check
                if 'embedding' in result:
                    # Check if it's a list of embeddings or single
                    if isinstance(result['embedding'][0], list) or isinstance(result['embedding'][0], np.ndarray):
                         embeddings_list.extend(result['embedding'])
                    else:
                         embeddings_list.append(result['embedding'])
            except Exception as e:
                print(f"Error embedding batch: {e}")
                # Fallback to single if batch fails? Or just re-raise. 
                # For demo, let's just try single for this batch or fail.
                raise e
        
        if not embeddings_list:
            return 0
        
        self.embeddings = np.array(embeddings_list).astype('float32')

        # 3. Indexing
        # Dynamically determine dimension from the first embedding
        current_dim = self.embeddings.shape[1]
        if self.index is None or self.index.d != current_dim:
            self.dimension = current_dim
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(self.embeddings)
        
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict], List[float], np.ndarray]:
        """
        Embeds the query and searches the FAISS index.
        Returns:
            - List of result dictionaries (text, score)
            - Query embedding (for visualization)
        """
        if not self.index:
            raise ValueError("Index not built. Please ingest data first.")
            
        # Embed query
        # Use task_type='retrieval_query' for queries
        query_result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array([query_result['embedding']]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "distance": float(distances[0][i])
                })
                
        return results, query_embedding[0]
