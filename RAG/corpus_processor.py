import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import json
from pathlib import Path
import re

class CorpusProcessor:
    """
    Processes a text corpus file into chunks for a RAG system.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        """
        Initialize the processor with a pre-trained model.

        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Loaded {model_name} on {device}")

    def load_corpus(self, file_path: str) -> str:
        """
        Load the corpus from a text file.

        Args:
            file_path: Path to the corpus file

        Returns:
            The loaded corpus text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_paragraph_chunks(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on newlines.

        Args:
            text: The text to split

        Returns:
            List of paragraphs
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_short_paragraphs(self, paragraphs: List[str], min_chars: int = 200) -> List[str]:
        """
        Merge short paragraphs to ensure chunks aren't too small.

        Args:
            paragraphs: List of paragraph strings
            min_chars: Minimum number of characters for a standalone paragraph

        Returns:
            List of merged paragraphs
        """
        merged = []
        current = ""

        for p in paragraphs:
            # If adding this paragraph would make current exceed min_chars
            # and current already has content, save current and start new
            if len(current) >= min_chars and current:
                merged.append(current)
                current = p
            # Otherwise append to current
            else:
                if current:
                    current += " " + p
                else:
                    current = p

        # Don't forget the last paragraph
        if current:
            merged.append(current)

        return merged

    def _split_long_paragraphs(self, paragraphs: List[str], max_chars: int = 1000) -> List[str]:
        """
        Split paragraphs that are too long.

        Args:
            paragraphs: List of paragraph strings
            max_chars: Maximum number of characters per paragraph

        Returns:
            List of split paragraphs
        """
        result = []

        for p in paragraphs:
            # If paragraph is short enough, keep as is
            if len(p) <= max_chars:
                result.append(p)
                continue

            # Otherwise split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current = ""

            for sentence in sentences:
                # If adding this sentence would exceed max_chars, save current and start new
                if len(current) + len(sentence) > max_chars and current:
                    result.append(current)
                    current = sentence
                # Otherwise append to current
                else:
                    if current:
                        current += " " + sentence
                    else:
                        current = sentence

            # Don't forget the last part
            if current:
                result.append(current)

        return result

    def chunk_corpus(self, text: str, min_chars: int = 200, max_chars: int = 1000) -> List[Dict[str, str]]:
        """
        Process the corpus into appropriately sized chunks.

        Args:
            text: The corpus text
            min_chars: Minimum characters per chunk
            max_chars: Maximum characters per chunk

        Returns:
            List of dictionaries with chunk IDs and text
        """
        # Initial split by paragraphs
        paragraphs = self._get_paragraph_chunks(text)
        print(f"Initial paragraph count: {len(paragraphs)}")

        # Merge short paragraphs
        merged = self._merge_short_paragraphs(paragraphs, min_chars)
        print(f"After merging short paragraphs: {len(merged)}")

        # Split long paragraphs
        chunks = self._split_long_paragraphs(merged, max_chars)
        print(f"Final chunk count: {len(chunks)}")

        # Create chunk dictionaries
        return [
            {
                'id': f"chunk_{i}",
                'text': chunk
            }
            for i, chunk in enumerate(chunks)
        ]

    def embed_chunks(self, chunks: List[Dict[str, str]], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Embed document chunks.

        Args:
            chunks: List of document chunks with 'id' and 'text' fields
            batch_size: Batch size for embedding generation

        Returns:
            Dictionary mapping chunk IDs to embedding vectors
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, batch_size=batch_size,
                                     show_progress_bar=True, convert_to_numpy=True)

        # Create a dictionary mapping chunk IDs to embeddings
        chunk_embeddings = {}
        for i, chunk in enumerate(chunks):
            chunk_embeddings[chunk['id']] = embeddings[i]

        return chunk_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        return self.model.encode(query, convert_to_numpy=True)

    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_path: str):
        """
        Save embeddings to disk.

        Args:
            embeddings: Dictionary mapping IDs to embeddings
            output_path: Path to save the embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {id: emb.tolist() for id, emb in embeddings.items()}

        with open(output_path, 'w') as f:
            json.dump(serializable_embeddings, f)

        print(f"Saved {len(embeddings)} embeddings to {output_path}")

    def save_chunks(self, chunks: List[Dict[str, str]], output_path: str):
        """
        Save chunk metadata to disk.

        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the chunks
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert chunks to a dictionary for easier lookup
        chunks_dict = {chunk['id']: chunk for chunk in chunks}

        with open(output_path, 'w') as f:
            json.dump(chunks_dict, f)

        print(f"Saved {len(chunks)} chunks to {output_path}")

    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk.

        Args:
            input_path: Path to load the embeddings from

        Returns:
            Dictionary mapping IDs to embedding vectors
        """
        with open(input_path, 'r') as f:
            serialized_embeddings = json.load(f)

        # Convert lists back to numpy arrays
        embeddings = {id: np.array(emb) for id, emb in serialized_embeddings.items()}

        print(f"Loaded {len(embeddings)} embeddings from {input_path}")
        return embeddings

    def load_chunks(self, input_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load chunks from disk.

        Args:
            input_path: Path to load the chunks from

        Returns:
            Dictionary mapping chunk IDs to chunk data
        """
        with open(input_path, 'r') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks
