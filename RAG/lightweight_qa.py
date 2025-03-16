import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM, AutoTokenizer

class LightweightQA:
    def __init__(self, model_name="google/flan-t5-small", device=None):
        """
        Initialize the QA system with a lightweight model
        
        Args:
            model_name: HuggingFace model name (default: google/flan-t5-small)
            device: Device to load the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if "t5" in model_name.lower() or "bart" in model_name.lower() or "flan" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model_type = "seq2seq"
        elif "distilbert" in model_name.lower() or "bert" in model_name.lower() or "roberta" in model_name.lower():
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.model_type = "extractive"
        elif "phi" in model_name.lower():  # Handle Phi-2 model
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model_type = "causal"
            # Fix for Phi-2: set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            # Default to seq2seq for other models
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model_type = "seq2seq"
        
        # Enable model evaluation mode for faster inference
        self.model.eval()
        
    def get_answer(self, question, chunks, max_length=150):
        """
        Get a concise answer based on the question and retrieved chunks.
    
        Args:
            question: The query string.
            chunks: List of relevant text chunks with similarity scores.
            max_length: Maximum length of generated answer.
    
        Returns:
            String containing the concise answer.
        """
        # Sort chunks by similarity score
        sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
    
        # Create a context from top chunks (limit context length)
        context = ""
        for chunk in sorted_chunks:
            context += f"{chunk.get('text', '')}\n"
            if len(context) > 1500:  # Limit context to avoid token limits
                break
    
        if self.model_type == "seq2seq":
            # For sequence-to-sequence models (T5, FLAN, BART, etc.)
            prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer the question concisely using only information from the context:"
            
            # Generate answer
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    min_length=10,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        elif self.model_type == "extractive":
            # For extractive QA models (BERT, DistilBERT, RoBERTa, etc.)
            inputs = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
            # Get start and end logits
            with torch.no_grad():
                outputs = self.model(**inputs)
    
            # Get most likely answer span
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            tokens = inputs["input_ids"][0][answer_start:answer_end]
            answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
    
        elif self.model_type == "causal":
            # For causal models (GPT, Phi-2, etc.)
            prompt = f"Answer the following question based ONLY on the information provided in the context. If the answer isn't in the context, answer based on any knowledge you might have or make an educated guess.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Calculate the prompt length to properly extract only the generated text
                prompt_length = inputs["input_ids"].shape[1]
                
                output_tokens = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    min_length=10,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                
                # Extract only the newly generated tokens (exclude the input prompt tokens)
                answer_tokens = output_tokens[0][prompt_length:]
                
                # Decode only the answer tokens
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                
            # Clean up any trailing explanations
            if "Explanation:" in answer:
                answer = answer.split("Explanation:", 1)[0].strip()
            
        else:
            raise ValueError("Unsupported model type")
    
        return answer
        
    def batch_get_answers(self, questions, chunks_list, max_length=150):
        """
        Process multiple questions in a batch for more efficient inference
        
        Args:
            questions: List of question strings
            chunks_list: List of chunk lists corresponding to each question
            max_length: Maximum length of generated answers
            
        Returns:
            List of answer strings
        """
        batch_size = len(questions)
        if batch_size == 0:
            return []
            
        batch_contexts = []
        
        # Prepare contexts for each question
        for chunks in chunks_list:
            # Sort chunks by similarity score
            sorted_chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
            
            # Create a context from top chunks (limit context length)
            context = ""
            for chunk in sorted_chunks:
                context += f"{chunk.get('text', '')}\n"
                if len(context) > 1500:  # Limit context to avoid token limits
                    break
                    
            batch_contexts.append(context)
            
        answers = []
        
        # Process based on model type
        if self.model_type == "seq2seq":
            # Prepare prompts
            prompts = [
                f"Question: {q}\n\nContext: {c}\n\nAnswer the question concisely using only information from the context:"
                for q, c in zip(questions, batch_contexts)
            ]
            
            # Tokenize (with padding)
            inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answers
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    min_length=10,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                
            # Decode all outputs
            for output in outputs:
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                answers.append(answer)
                
        elif self.model_type == "extractive":
            # Process questions individually for extractive models
            # (harder to batch efficiently due to different context lengths)
            for question, context in zip(questions, batch_contexts):
                inputs = self.tokenizer(
                    question,
                    context,
                    add_special_tokens=True,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get most likely answer span
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                tokens = inputs["input_ids"][0][answer_start:answer_end]
                answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
                answers.append(answer)
                
        # Update this section in the batch_get_answers method for causal models
        elif self.model_type == "causal":
            # Prepare prompts
            prompts = [
                f"Answer the following question based ONLY on the information provided in the context. If the answer isn't in the context, answer based on any knowledge you might have or make an educated guess.\n\nQuestion: {q}\n\nContext: {c}\n\nAnswer:"
                for q, c in zip(questions, batch_contexts)
            ]
            
            # Process in smaller batches to avoid OOM errors for large models
            sub_batch_size = 2  # Adjust based on your GPU memory
            sub_answers = []
            
            for i in range(0, len(prompts), sub_batch_size):
                sub_prompts = prompts[i:i+sub_batch_size]
                inputs = self.tokenizer(sub_prompts, return_tensors="pt",
                                      truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Store the input lengths to extract only the new tokens later
                    input_lengths = [len(ids) for ids in inputs["input_ids"]]
                    
                    output_tokens = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_length,
                        min_length=10,
                        num_beams=4,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                
                    # Process each output in the batch
                    for j, output in enumerate(output_tokens):
                        # Extract only the newly generated tokens
                        answer_tokens = output[input_lengths[j % len(input_lengths)]:]
                        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                        
                        # Clean up explanations
                        if "Explanation:" in answer:
                            answer = answer.split("Explanation:", 1)[0].strip()
                            
                        sub_answers.append(answer)
            
            answers = sub_answers
            
        else:
            raise ValueError("Unsupported model type")
            
        return answers