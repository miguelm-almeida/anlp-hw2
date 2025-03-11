import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class DocumentStore:
    def __init__(self, processed_data_path='processed_data/processed_documents.json', 
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 db_path='chroma_db'):
        self.processed_data_path = processed_data_path
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(persist_directory=db_path))
        self.collection = self.client.get_or_create_collection("pittsburgh_cmu_docs")
        
    def load_documents(self):
        """Load documents from processed data file"""
        with open(self.processed_data_path, 'r') as f:
            documents = json.load(f)
        return documents
    
    def index_documents(self, documents=None):
        """Index documents in the vector store"""
        if documents is None:
            documents = self.load_documents()
            
        print(f"Indexing {len(documents)} documents...")
        
        # Prepare data for batch indexing
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [{"source": doc["source"]} for doc in documents]
        
        # Add documents to the collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            batch_ids = ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            print(f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
        print("Indexing complete!")
    
    def retrieve_documents(self, query, top_k=5):
        """Retrieve relevant documents for a query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_docs = []
        for i, doc in enumerate(results['documents'][0]):
            retrieved_docs.append({
                'text': doc,
                'source': results['metadatas'][0][i]['source'],
                'id': results['ids'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            })
            
        return retrieved_docs

if __name__ == "__main__":
    doc_store = DocumentStore()
    doc_store.index_documents()
    
    # Test retrieval
    query = "When was Carnegie Mellon University founded?"
    docs = doc_store.retrieve_documents(query)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1} (Score: {doc['score']}):")
        print(f"Source: {doc['source']}")
        print(f"Text: {doc['text'][:200]}...") 