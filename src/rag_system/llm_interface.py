from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LLMInterface:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # For memory efficiency on consumer GPUs
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        print("Model loaded successfully!")
    
    def generate_answer(self, question, documents):
        """Generate an answer for a question using the provided documents"""
        # Construct prompt with retrieved documents
        context = "\n\n".join([doc["text"] for doc in documents])
        
        # Format prompt according to model's expected format
        if "llama" in self.model_name.lower():
            # Llama 2 specific prompt format
            prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant that answers questions about Pittsburgh and CMU.
Answer the question based ONLY on the following context.
If you don't know the answer based on the context, simply say "I don't have enough information to answer."
Be concise and to the point.
<</SYS>>

Context:
{context}

Question: {question} [/INST]"""
        else:
            # Generic prompt format
            prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        # Generate answer
        response = self.pipe(prompt)[0]["generated_text"]
        
        # Extract answer from the response
        if "llama" in self.model_name.lower():
            # Llama 2 specific response parsing
            answer = response.split("[/INST]")[-1].strip()
        else:
            # Generic response parsing
            answer = response[len(prompt):].strip()
        
        return answer

if __name__ == "__main__":
    # Test with sample documents
    llm = LLMInterface()
    
    question = "When was Carnegie Mellon University founded?"
    documents = [
        {"text": "Carnegie Mellon University was founded in 1900 by Andrew Carnegie as the Carnegie Technical Schools."}
    ]
    
    answer = llm.generate_answer(question, documents)
    print(f"Question: {question}")
    print(f"Answer: {answer}") 