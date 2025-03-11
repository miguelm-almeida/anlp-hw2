import os
import pypdf
import json

class PDFProcessor:
    def __init__(self, output_dir='raw_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path, output_filename):
        """Extract text from a PDF and save to file"""
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Extract text using pypdf
            reader = pypdf.PdfReader(pdf_path)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            # Save to file
            with open(os.path.join(self.output_dir, output_filename), 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"Successfully extracted text to {output_filename}")
            return True
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return False
    
    def process_pdf_list(self, pdf_list_file):
        """Process PDFs from a list stored in a file"""
        with open(pdf_list_file, 'r') as f:
            pdf_paths = [line.strip() for line in f if line.strip()]
            
        results = []
        for i, pdf_path in enumerate(pdf_paths):
            filename = f"pdf_{i:04d}.txt"
            success = self.extract_text_from_pdf(pdf_path, filename)
            results.append({
                "pdf_path": pdf_path,
                "filename": filename,
                "success": success
            })
            
        # Save metadata
        with open(os.path.join(self.output_dir, "pdf_metadata.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Completed processing {len(pdf_paths)} PDFs")

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdf_list("pdfs_to_process.txt") 