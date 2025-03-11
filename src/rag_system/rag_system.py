import os
import json
from document_store import DocumentStore
from llm_interface import LLMInterface

class RAGSystem:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.doc_store = DocumentStore()
        self.llm = LLMInterface(model_name)
    
    def answer_question(self, question, top_k=5):
        """Answer a question using RAG"""
        # Retrieve relevant documents
        documents = self.doc_store.retrieve_documents(question, top_k=top_k)
        
        # Generate answer
        answer = self.llm.generate_answer(question, documents)
        
        return {
            "question": question,
            "answer": answer,
            "documents": documents
        }
    
    def process_questions_file(self, questions_file, output_file):
        """Process a file of questions and save answers to output file"""
        # Read questions
        with open(questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(questions)} questions...")
        
        # Process each question
        results = {}
        for i, question in enumerate(questions):
            print(f"Question {i+1}/{len(questions)}: {question}")
            
            response = self.answer_question(question)
            results[str(i+1)] = response["answer"]
            
            print(f"Answer: {response['answer']}\n")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the RAG system on a questions file")
    parser.add_argument("--questions", required=True, help="Path to questions file")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Name of the HuggingFace model to use")
    
    args = parser.parse_args()
    
    rag = RAGSystem(model_name=args.model)
    rag.process_questions_file(args.questions, args.output) 