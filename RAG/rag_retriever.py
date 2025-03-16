import numpy as np
import json
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import os
import torch
from dotenv import load_dotenv
from corpus_processor import CorpusProcessor

class RAGRetriever:
    """
    A retriever that uses embeddings to find relevant chunks for a query.
    """
    def __init__(self, chunks_path: str, embeddings_path: str):
        """
        Initialize the retriever.
        
        Args:
            chunks_path: Path to the chunks JSON file
            embeddings_path: Path to the embeddings JSON file
        """
        # Load chunks and embeddings
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)
        
        with open(embeddings_path, 'r') as f:
            serialized_embeddings = json.load(f)
            # Convert lists back to numpy arrays with consistent dtype
            self.embeddings = {id: np.array(emb, dtype=np.float32) for id, emb in serialized_embeddings.items()}
        
        print(f"Loaded {len(self.chunks)} chunks and {len(self.embeddings)} embeddings")
        
        # Initialize processor for query embedding
        self.processor = CorpusProcessor()
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of the most relevant chunks with their similarity scores
        """
        # Get query embedding and ensure it's float32
        query_embedding = self.processor.embed_query(query).astype(np.float32)
        
        # Calculate similarity with all chunk embeddings using manual cosine similarity
        similarities = []
        for chunk_id, embedding in self.embeddings.items():
            # Manual cosine similarity calculation
            dot_product = np.dot(query_embedding, embedding)
            query_norm = np.linalg.norm(query_embedding)
            embedding_norm = np.linalg.norm(embedding)
            
            # Avoid division by zero
            if query_norm == 0 or embedding_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (query_norm * embedding_norm)
            
            similarities.append({
                'id': chunk_id,
                'similarity': similarity,
                'text': self.chunks[chunk_id]['text']
            })
        
        # Sort by similarity (highest first)
        sorted_chunks = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k chunks
        return sorted_chunks[:top_k]