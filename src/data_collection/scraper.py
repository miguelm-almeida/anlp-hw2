import os
import requests
from bs4 import BeautifulSoup
import json
import time

class WebScraper:
    def __init__(self, output_dir='raw_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def scrape_webpage(self, url, output_filename):
        """Scrape content from a webpage and save to file"""
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (customize based on website structure)
            content = soup.find('main') or soup.find('article') or soup.find('body')
            
            # Clean content (remove scripts, styles, etc.)
            for tag in content.find_all(['script', 'style', 'nav', 'footer']):
                tag.extract()
                
            # Extract text
            text = content.get_text(separator='\n', strip=True)
            
            # Save to file
            with open(os.path.join(self.output_dir, output_filename), 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"Successfully saved content to {output_filename}")
            
            # Be nice to the server
            time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return False
            
    def scrape_from_url_list(self, url_list_file):
        """Scrape content from a list of URLs stored in a file"""
        with open(url_list_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
            
        results = []
        for i, url in enumerate(urls):
            filename = f"doc_{i:04d}.txt"
            success = self.scrape_webpage(url, filename)
            results.append({
                "url": url,
                "filename": filename,
                "success": success
            })
            
        # Save metadata
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Completed scraping {len(urls)} URLs")

if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrape_from_url_list("urls_to_scrape.txt") 