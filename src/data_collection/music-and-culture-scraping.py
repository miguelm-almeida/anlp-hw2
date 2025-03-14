import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from collections import deque
import os
import re
import unicodedata


def extract_text_content(url):
    """
    Extract all text content from a web page using Beautiful Soup

    Args:
        url (str): The URL of the page to scrape

    Returns:
        tuple: (text_content, page_title, links)

    Note:
        Normalizes all line endings to Unix-style '\n' and removes unusual line terminators
    """
    try:
        # Send request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse HTML with Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract page title
        title = soup.title.text.strip() if soup.title else "No title found"

        # Extract all text
        # Remove script and style elements that might contain text not meant for display
        for script_or_style in soup(['script', 'style', 'meta', 'noscript']):
            script_or_style.decompose()

        # Get all text
        text_content = []

        # Add the title first - ensure no unusual characters in title
        clean_title = unicodedata.normalize('NFKD', title)
        clean_title = re.sub(r'[\r\n\u2028\u2029]', ' ', clean_title)  # Replace line breaks with spaces

        # Extract headings with hierarchy
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(tag):
                if heading.text.strip():
                    # Add heading with appropriate level indicator
                    level_indicator = "#" * int(tag[1])
                    # Clean heading text of unusual line terminators
                    clean_heading = unicodedata.normalize('NFKD', heading.text.strip())
                    clean_heading = re.sub(r'[\r\n\u2028\u2029]', ' ', clean_heading)
                    # text_content.append(f"{level_indicator} {clean_heading}\n\n")

        # Extract paragraphs
        for p in soup.find_all('p'):
            if p.text.strip():
                # Clean paragraph text of unusual line terminators
                clean_para = unicodedata.normalize('NFKD', p.text.strip())
                clean_para = re.sub(r'[\r\n\u2028\u2029]', ' ', clean_para)
                text_content.append(f"{clean_para}\n\n")

        # Get content from div elements that might contain text
        for div in soup.find_all('div'):
            # Only include direct text in divs (not text from child elements)
            direct_text = div.find(string=True, recursive=False)
            if direct_text and direct_text.strip():
                # Clean direct text of unusual line terminators
                clean_text = unicodedata.normalize('NFKD', direct_text.strip())
                clean_text = re.sub(r'[\r\n\u2028\u2029]', ' ', clean_text)
                text_content.append(f"{clean_text}\n\n")

        # Extract all links
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                links.append({
                    'text': link.text.strip(),
                    'url': href
                })

        # Join all text content
        full_text = "".join(text_content)

        # Normalize line endings and remove unusual line terminators
        # Replace Windows line endings (\r\n) with Unix line endings (\n)
        full_text = full_text.replace('\r\n', '\n')
        # Replace old Mac line endings (\r) with Unix line endings (\n)
        full_text = full_text.replace('\r', '\n')
        # Remove any other unusual line terminators like Line Separator (LS) or Paragraph Separator (PS)
        full_text = full_text.replace('\u2028', '\n')  # Line Separator (LS)
        full_text = full_text.replace('\u2029', '\n')  # Paragraph Separator (PS)
        # Remove any other control characters except normal whitespace
        full_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', full_text)

        return full_text, title, links
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return f"Error scraping {url}: {e}", "Error", []


def normalize_url(base_url, href):
    """
    Normalize a URL by resolving relative paths and ensuring it's from the same domain

    Args:
        base_url (str): The base URL for resolving relative paths
        href (str): The URL or path to normalize

    Returns:
        str or None: The normalized URL if valid, None otherwise
    """
    try:
        # Parse the base URL to get the domain
        parsed_base = urllib.parse.urlparse(base_url)
        base_domain = parsed_base.netloc

        # Skip if it's a fragment, javascript, mailto, or other non-HTTP scheme
        if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
            return None

        # Resolve the URL (handles relative paths)
        full_url = urllib.parse.urljoin(base_url, href)
        parsed_url = urllib.parse.urlparse(full_url)

        # Only return URLs from the same domain and with HTTP/HTTPS scheme
        if parsed_url.netloc == base_domain and parsed_url.scheme in ['http', 'https']:
            # Remove fragments
            return full_url.split('#')[0]
        return None
    except Exception:
        return None


def create_filename_from_url(url, title):
    """
    Create a safe filename from a URL and title

    Args:
        url (str): The URL
        title (str): The page title

    Returns:
        str: A safe filename
    """
    # Extract domain name
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc

    # Extract path and remove trailing slash if present
    path = parsed_url.path.rstrip('/')

    # Replace special characters in the path with underscores
    path = re.sub(r'[^a-zA-Z0-9]', '_', path)

    # Clean the title
    clean_title = re.sub(r'[^a-zA-Z0-9]', '_', title)
    clean_title = re.sub(r'_+', '_', clean_title)  # Replace multiple underscores with a single one
    clean_title = clean_title[:50]  # Limit title length

    # Build the filename
    if path:
        filename = f"{domain}{path}_{clean_title}.txt"
    else:
        filename = f"{domain}_{clean_title}.txt"

    # Ensure filename doesn't have double underscores
    filename = re.sub(r'_+', '_', filename)

    # Ensure filename isn't excessively long
    if len(filename) > 100:
        filename = filename[:100] + ".txt"

    return filename


def crawl_website(start_url, max_pages=100):
    """
    Crawl a website starting from a given URL, up to a maximum number of pages

    Args:
        start_url (str): The URL to start crawling from
        max_pages (int): Maximum number of pages to crawl

    Returns:
        list: List of dictionaries containing data from all crawled pages
    """
    # Parse the start URL to get the domain
    parsed_url = urllib.parse.urlparse(start_url)
    domain = parsed_url.netloc

    # Create a directory for the results if it doesn't exist
    domain_dir = domain.replace(':', '_').replace('/', '_')
    output_dir = f'raw-data/music-and-culture/{domain_dir}'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the queue with the start URL
    queue = deque([start_url])

    # Set to keep track of visited URLs
    visited = set([start_url])

    # Counter for the number of pages crawled
    pages_crawled = 0

    print(f"Starting crawl of {domain} from {start_url}")

    while queue and pages_crawled < max_pages:
        # Get the next URL from the queue
        current_url = queue.popleft()

        print(f"Crawling {current_url} ({pages_crawled + 1}/{max_pages})")

        # Extract content from the page
        text_content, title, links = extract_text_content(current_url)

        if text_content:
            # Create a filename for this page
            filename = create_filename_from_url(current_url, title)

            # Save the text content to a file with standardized line endings
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8', newline='\n') as f:
                # f.write(f"TITLE: {title}\nURL: {current_url}\n")
                # f.write("-" * 50 + "\n")
                f.write(text_content)

            print(f"Saved text to {output_dir}/{filename}")

            # Increment the counter
            pages_crawled += 1

            # Add new links to the queue
            for link in links:
                href = link['url']
                normalized_url = normalize_url(current_url, href)

                if normalized_url and normalized_url not in visited:
                    queue.append(normalized_url)
                    visited.add(normalized_url)

        # Be nice to the server
        time.sleep(0.01)

    print(f"Finished crawling {domain}, visited {pages_crawled} pages")
    return pages_crawled


def crawl_multiple_sites(urls, max_pages_per_site=50):
    """
    Crawl multiple websites from a list of starter URLs

    Args:
        urls (list): List of URLs to start crawling from
        max_pages_per_site (int): Maximum number of pages to crawl per site

    Returns:
        dict: Dictionary mapping domains to their crawled content
    """
    # Create main directory for results
    os.makedirs('raw-data/music-and-culture', exist_ok=True)

    results = {}

    for url in urls:
        try:
            # Parse the URL to get the domain
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc

            print(f"\n{'='*40}\nStarting crawl of {domain}\n{'='*40}")

            # Crawl the website
            pages_crawled = crawl_website(url, max_pages=max_pages_per_site)

            # Store the results
            results[domain] = pages_crawled

            print(f"Crawled {pages_crawled} pages from {domain}")
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    return results


# Define your list of URLs to crawl here
urls_to_crawl = [
    "https://www.pittsburghsymphony.org/",
    "https://pittsburghopera.org/",
    "https://trustarts.org/",
    "https://carnegiemuseums.org/",
    "https://www.heinzhistorycenter.org/",
    "https://www.thefrickpittsburgh.org/",
    "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
    "https://www.picklesburgh.com/",
    "https://www.pghtacofest.com/",
    "https://pittsburghrestaurantweek.com/",
    "https://littleitalydays.com/",
    "https://bananasplitfest.com/"
]

# Set the maximum number of pages to crawl per site
max_pages_per_site = 50

# Create the base directory structure
os.makedirs('raw-data', exist_ok=True)
os.makedirs('raw-data/music-and-culture', exist_ok=True)

# Call the function with your list of URLs
results = crawl_multiple_sites(urls_to_crawl, max_pages_per_site=max_pages_per_site)

print("\nCrawl complete!")
print("Summary of pages crawled:")
for domain, count in results.items():
    print(f"  - {domain}: {count} pages")
print(f"Results saved to the 'raw-data/music-and-culture' directory")