import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK resources
nltk.download('punkt')

class DocumentPreprocessor:
    def __init__(self, raw_data_dir='raw_data', processed_dir='processed_data'):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def clean_text(self, text):
        """Clean the text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Additional cleaning can be added as needed
        
        return text.strip()
    
    def segment_into_chunks(self, text, max_length=512):
        """Split the text into manageable chunks, preserving sentence boundaries"""
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def process_documents(self):
        """Process all documents in the raw data directory"""
        documents = []
        
        # Process all text files in the raw data directory
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.raw_data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Clean the text
                    cleaned_text = self.clean_text(text)
                    
                    # Segment into chunks
                    chunks = self.segment_into_chunks(cleaned_text)
                    
                    # Add to documents list
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{filename.split('.')[0]}_chunk_{i}"
                        documents.append({
                            "id": doc_id,
                            "text": chunk,
                            "source": filename
                        })
                    
                    print(f"Processed {filename} into {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save processed documents
        with open(os.path.join(self.processed_dir, 'processed_documents.json'), 'w') as f:
            json.dump(documents, f, indent=2)
            
        print(f"Saved {len(documents)} processed document chunks")

if __name__ == "__main__":
    processor = DocumentPreprocessor()
    processor.process_documents() 