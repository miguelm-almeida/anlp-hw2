import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from rag_retriever import RAGRetriever
from rag_question_answerer import RAGQuestionAnswerer
from lightweight_qa import LightweightQA  # Make sure to use the optimized version

def process_single_query(qa_system, question, qa, retriever, verbose=False):
    """Process a single question and return the answer"""
    # Get RAG results
    result = qa.answer(question, top_k=3)
    chunks = result["chunks"]
    
    if verbose:
        print("\nQuestion:", question)
        print("\nRetrieved chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} (similarity: {chunk['similarity']:.4f}):")
            print(chunk["text"])
        print(f"\n=== Using {qa_system.__class__.__name__} ===")
    
    # Get answer from model (using the pre-loaded model)
    answer = qa_system.get_answer(question, chunks)
    
    if verbose:
        print(f"Answer: {answer}")
    
    return answer

def batch_process_queries(model_type, questions_list, batch_size=4):
    """Process multiple questions in batches with shared resources"""
    # File paths
    chunks_path = "data/chunks.json"
    embeddings_path = "data/embeddings.json"
    
    # Model mapping
    model_mapping = {
        "flan-t5": "google/flan-t5-small",
        "distilbert": "distilbert/distilbert-base-uncased-distilled-squad",
        "phi-2": "microsoft/phi-2",
        "roberta": "deepset/roberta-base-squad2"
    }
    selected_model = model_mapping[model_type]
    
    # Create retriever and QA system (shared across all queries)
    retriever = RAGRetriever(chunks_path, embeddings_path)
    qa = RAGQuestionAnswerer(retriever)
    
    # Check for GPU availability and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize QA system once
    print(f"Loading model: {selected_model}")
    qa_system = LightweightQA(model_name=selected_model, device=device)
    
    # Set torch to inference mode for better performance
    torch.set_grad_enabled(False)
    
    # Process in batches
    answers = []
    
    for i in tqdm(range(0, len(questions_list), batch_size), desc="Processing Batches"):
        batch_questions = questions_list[i:i+batch_size]
        
        # Retrieve chunks for all questions in batch
        batch_chunks = []
        for question in batch_questions:
            result = qa.answer(question, top_k=3)
            batch_chunks.append(result["chunks"])
        
        # Get answers using the batch method for efficiency
        batch_answers = qa_system.batch_get_answers(batch_questions, batch_chunks)
        answers.extend(batch_answers)
        
        # Print first result as sample
        if i == 0:
            print(f"\nSample question: {batch_questions[0]}")
            print(f"Sample answer: {batch_answers[0]}\n")
    
    return answers

def main():
    parser = argparse.ArgumentParser(description="Run RAG QA with a selected model.")
    parser.add_argument(
        "--model",
        choices=["flan-t5", "distilbert", "phi-2", "roberta"],
        required=True,
        help="Choose a model: flan-t5, distilbert, phi-2, roberta"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Enter the question to be answered."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path to CSV file containing questions in a 'question' column"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple questions"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use FP16 precision to reduce memory usage and improve speed"
    )
    
    args = parser.parse_args()
    
    # File paths
    chunks_path = "data/chunks.json"
    embeddings_path = "data/embeddings.json"
    
    # Model selection
    model_mapping = {
        "flan-t5": "google/flan-t5-small",
        "distilbert": "distilbert/distilbert-base-uncased-distilled-squad",
        "phi-2": "microsoft/phi-2",
        "roberta": "deepset/roberta-base-squad2"
    }
    selected_model = model_mapping[args.model]
    
    # Check for GPU availability and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize QA system once with the selected model
    print(f"Loading model: {selected_model}")
    qa_system = LightweightQA(model_name=selected_model, device=device)
    
    # Use half precision if requested (only works on CUDA)
    if args.half_precision and device == "cuda":
        print("Using half precision (FP16)")
        qa_system.model = qa_system.model.half()
    
    # Create retriever and QA system
    retriever = RAGRetriever(chunks_path, embeddings_path)
    qa = RAGQuestionAnswerer(retriever)
    
    # Check if we're processing a single question or a CSV file
    if args.question:
        # Process single question with verbose output
        answer = process_single_query(qa_system, args.question, qa, retriever, verbose=True)
        
    elif args.csv_file:
        import pandas as pd
        
        # Read questions from CSV
        df = pd.read_csv(args.csv_file)
        if "question" not in df.columns:
            raise ValueError("CSV file must contain a column named 'question'.")
        
        questions_list = df["question"].tolist()
        print(f"Processing {len(questions_list)} questions...")
        
        # Process all questions in batches
        answers = batch_process_queries(args.model, questions_list, args.batch_size)
        
        # Save results
        df["output"] = answers
        output_file = args.csv_file.replace('.csv', '_with_outputs.csv')
        df.to_csv(output_file, index=False)
        print(f"Inference completed. Results saved to '{output_file}'.")
        
    else:
        parser.error("Either --question or --csv_file must be provided")

if __name__ == "__main__":
    main()