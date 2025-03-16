import pandas as pd
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def run_inference(model_name, csv_file, batch_size=8):
    """Runs inference on a CSV file using a specified model with GPU acceleration."""
    # Model name mapping
    model_mappings = {'google/flan-t5-large': 'google/flan-t5-large'}
    full_model_name = model_mappings.get(model_name, model_name)
    
    print(f"Loading model: {full_model_name}")
    
    # Load model and tokenizer once (will use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with optimizations for inference
    model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    
    # Enable evaluation mode for better inference performance
    model.eval()
    
    # Load and prepare data
    df = pd.read_csv(csv_file)
    if "question" not in df.columns:
        raise ValueError("CSV file must contain a column named 'question'.")
    
    questions = df["question"].tolist()
    results = []
    
    # Process in batches to utilize GPU effectively
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):
            batch_questions = questions[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Generate answers
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode and add to results
            batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Print a sample of results for debugging
            if i == 0:
                print(f"\nSample question: {batch_questions[0]}")
                print(f"Sample answer: {batch_results[0]}\n")
            
            results.extend(batch_results)
    
    # Add results to DataFrame and save
    df["output"] = results
    output_file = csv_file.replace('.csv', '_with_outputs.csv')
    df.to_csv(output_file, index=False)
    print(f"Inference completed. Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV file with a given model using GPU acceleration.")
    parser.add_argument("--model", required=True, help="Model to use for inference")
    parser.add_argument("--csv_file", required=True, help="CSV file containing queries")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()
    
    run_inference(args.model, args.csv_file, args.batch_size)