@echo off
echo Setting up RAG system...

:: Create necessary directories
mkdir chroma_db 2>nul

:: Index documents in vector store
echo Indexing documents...
python src/rag_system/document_store.py

echo System setup complete! 