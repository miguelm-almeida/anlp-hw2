from typing import List, Dict, Optional
from rag_retriever import RAGRetriever

class RAGQuestionAnswerer:
    """
    A question answerer that uses RAG to generate answers.
    """
    def __init__(self, retriever: RAGRetriever, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the question answerer.
        
        Args:
            retriever: The retriever to use
            api_key: OpenAI API key (will use environment variable if not provided)
            model: Model to use for answer generation
        """
        self.retriever = retriever
    
    def answer(self, question: str, top_k: int = 3) -> Dict:
        """
        Generate an answer for a question using RAG.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with the answer and the retrieved chunks
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        # Prepare context for the model
        context = "\n\n".join([chunk["text"] for chunk in chunks])
        return {
            "question": question,
            "chunks": chunks
        }