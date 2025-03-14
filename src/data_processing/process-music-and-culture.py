import os
import re
import glob
from pathlib import Path
import argparse

def clean_text(text):
    """
    Clean the text by removing extra whitespace, redundant newlines,
    and other unwanted content.
    """
    # Remove TITLE and URL patterns
    text = re.sub(r'TITLE:.*?URL:.*?--+', '', text)
    
    # Remove markdown headers (# and ## lines)
    text = re.sub(r'#+\s+.*?\n', '', text)
    
    # Remove navigation markers and UI elements
    text = re.sub(r'## Navigation.*?##', '', text)
    text = re.sub(r'## Plan Your Trip.*?##', '', text)
    text = re.sub(r'## Experience Builder.*?##', '', text)
    text = re.sub(r'## Quick Search.*?##', '', text)
    text = re.sub(r'## Location Map', '', text)
    
    # Remove partner references and IDs
    text = re.sub(r'/\d+/VP:FEATURED-PARTNER-\d+', '', text)
    
    # Remove footer information
    text = re.sub(r'\d+ Fifth Avenue.*?Pittsburgh, PA \d+', '', text)
    text = re.sub(r'\(\d+\) \d+-\d+.*?Toll Free: \(\d+\) \d+-\d+', '', text)
    
    # Remove social media markers
    text = re.sub(r'Share Your Story On Social #\w+', '', text)
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    
    # Remove common boilerplate texts
    text = re.sub(r'Cookie Policy|Privacy Policy|Terms of Use|©.*?reserved', '', text)
    text = re.sub(r'Subscribe|Sign up|Newsletter|Log in|Sign in', '', text)
    text = re.sub(r'This site uses cookies.*?experience\.', '', text)
    text = re.sub(r'©️\d+ Visit Pittsburgh\. All Rights Reserved\.', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove social media references
    text = re.sub(r'(?i)follow us on|facebook|twitter|instagram|linkedin', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    
    # Remove JavaScript and CSS snippets
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove numbers and text patterns that appear to be identifiers
    text = re.sub(r'\b\d+ (Saved|0)\b', '', text)
    
    # Clean up multiple newlines to single newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_files(input_dir, output_file):
    """
    Process all files in the input directory and write clean content
    to the output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize a counter for processed files
    file_count = 0
    processed_content = []
    
    # Get all files in the directory and its subdirectories
    for filepath in glob.glob(os.path.join(input_dir, '**', '*.*'), recursive=True):
        # Skip directories and non-text files
        if os.path.isdir(filepath) or not is_text_file(filepath):
            continue
            
        try:
            # Read and process the file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean the content
            clean_content = clean_text(content)
            
            # Skip if no meaningful content left after cleaning
            if len(clean_content.strip()) < 50:
                continue
            
            # Add to our collection of processed content
            processed_content.append(clean_content)
            file_count += 1
            
            if file_count % 10 == 0:
                print(f"Processed {file_count} files...")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Write all content to output file
    with open(output_file, 'w', encoding='utf-8') as outf:
        outf.write("\n\n".join(processed_content))
    
    print(f"Completed processing {file_count} files. Output saved to {output_file}")

def is_text_file(filepath):
    """
    Check if a file is likely to be a text file based on extension
    or content.
    """
    text_extensions = {'.txt', '.html', '.htm', '.md', '.csv', '.json', '.xml'}
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in text_extensions:
        return True
        
    # For files without a recognized extension, check the content
    try:
        with open(filepath, 'rb') as f:
            sample = f.read(1024)
        # Check if the content is mostly ASCII
        return is_binary_string(sample) is False
    except:
        return False

def is_binary_string(bytes_data):
    """
    Check if a string is binary (vs text).
    """
    # A simple heuristic: if more than 30% of the bytes are non-text,
    # we consider it binary
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
    return bool(bytes_data.translate(None, textchars))

def main():
    parser = argparse.ArgumentParser(description='Process and clean text files from music and culture directory.')
    parser.add_argument('--output', default='clean-data/music-and-culture.txt',
                        help='Path to the output file (default: clean-data/music-and-culture.txt)')
    parser.add_argument('--input', default='raw-data/music-and-culture',
                        help='Path to the input directory (default: raw-data/music-and-culture)')
    
    args = parser.parse_args()
    
    # Process the files
    process_files(args.input, args.output)

if __name__ == '__main__':
    main() 