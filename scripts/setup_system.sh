#!/bin/bash
echo "Setting up RAG system..."

# Create necessary directories
mkdir -p chroma_db

# Index documents in vector store
echo "Indexing documents..."
python src/rag_system/document_store.py

echo "System setup complete!" 