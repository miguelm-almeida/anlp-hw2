import pandas as pd
import argparse
import torch
from tqdm import tqdm
import os
import sys
import json
from lightweight_qa import LightweightQA

def run_closed_book_inference(model_name, model_type, questions, output_file, batch_size=4):
    """
    Run inference on questions in closed book mode (without RAG context)
    
    Args:
        model_name: HuggingFace model name
        model_type: Model type (flan-t5, distilbert, phi-2)
        questions: List of questions
        output_file: Path to save results
        batch_size: Batch size for processing
    """
    # Check for GPU availability and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize QA system
    print(f"Loading model: {model_name}")
    qa_system = LightweightQA(model_name=model_name, device=device)
    
    # Set torch to inference mode for better performance
    torch.set_grad_enabled(False)
    
    # Process in batches
    answers = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):
        batch_questions = questions[i:i+batch_size]
        
        # For closed book, we provide empty context
        empty_chunks = [[{"text": "", "similarity": 0.0}] for _ in range(len(batch_questions))]
        
        # Get answers
        if len(batch_questions) == 1:
            # Single question mode
            answer = qa_system.get_answer(batch_questions[0], empty_chunks[0])
            answers.append(answer)
        else:
            # Batch mode
            batch_answers = qa_system.batch_get_answers(batch_questions, empty_chunks)
            answers.extend(batch_answers)
        
        # Print first result as sample
        if i == 0:
            print(f"\nSample question: {batch_questions[0]}")
            print(f"Sample answer: {answers[0]}\n")
    
    # Create DataFrame and save results
    df = pd.DataFrame({
        'question': questions,
        'output': answers
    })
    
    df.to_csv(output_file, index=False)
    print(f"Inference completed. Results saved to '{output_file}'.")
    
    return answers

def main():
    parser = argparse.ArgumentParser(description="Run closed book QA evaluation.")
    parser.add_argument(
        "--model",
        choices=["flan-t5", "distilbert", "phi-2", "all"],
        required=True,
        help="Choose a model or 'all' to evaluate all models"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="../test/test_questions.txt",
        help="Path to text file with questions"
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        default="../test/test_answers.txt",
        help="Path to text file with ground truth answers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="closed_book_results",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    # Model mapping - using the same model names as in main.py
    model_mapping = {
        "flan-t5": "google/flan-t5-small",
        "distilbert": "distilbert/distilbert-base-uncased-distilled-squad",
        "phi-2": "microsoft/phi-2"
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load questions and answers
    with open(args.questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f.readlines()]
    
    with open(args.answers_file, 'r', encoding='utf-8') as f:
        ground_truth = [line.strip() for line in f.readlines()]
    
    # Create a merged dataframe with questions and ground truth answers
    df_truth = pd.DataFrame({
        'question': questions,
        'correct_answer': ground_truth
    })
    
    # Determine which models to evaluate
    models_to_evaluate = list(model_mapping.keys()) if args.model == "all" else [args.model]
    
    for model_type in models_to_evaluate:
        model_name = model_mapping[model_type]
        output_file = os.path.join(args.output_dir, f"{model_type}_closed_book.csv")
        
        print(f"\n=== Running closed book evaluation for {model_type} ===")
        answers = run_closed_book_inference(model_name, model_type, questions, output_file, args.batch_size)
        
        # Create merged file with predictions and ground truth for evaluation
        df_results = pd.DataFrame({
            'question': questions,
            'output': answers,
            'correct_answer': ground_truth
        })
        
        merged_file = os.path.join(args.output_dir, f"merged_qa_{model_type}_closed_book.csv")
        df_results.to_csv(merged_file, index=False)
        print(f"Merged file created at: {merged_file}")

if __name__ == "__main__":
    main()