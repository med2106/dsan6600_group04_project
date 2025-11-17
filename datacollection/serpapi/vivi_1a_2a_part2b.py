"""
Hair Type Image Scraper - Multi-Query Version
Uses multiple search variations to collect more images
(Pagination alone doesn't work well with Google Images API)
"""

import os
import requests
from serpapi import GoogleSearch
import time
from pathlib import Path
import hashlib
from urllib.parse import urlparse

class HairTypeImageScraperV2:
    def __init__(self, api_key, output_dir="hair_type_dataset"):
        """
        Initialize the scraper
        
        Args:
            api_key: Your SerpAPI key
            output_dir: Directory to save images
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define search query variations for each hair type
        # Using multiple queries gets us more unique images than pagination alone
        self.hair_type_queries = {
            "1a": [
                "1a straight hair",
                "type 1a hair",
                "fine straight hair 1a",
                "1a straight hair texture",
                "straight hair type 1a examples",
            ],
            "1b": [
                "1b straight hair",
                "type 1b hair",
                "1b hair texture",
                "straight hair 1b examples",
                "1b straight hair pattern",
            ],
            "1c": [
                "1c straight hair",
                "type 1c hair",
                "coarse straight hair 1c",
                "1c hair texture",
                "straight hair 1c examples",
            ],
            "2a": [
                "2a wavy hair",
                "type 2a hair",
                "loose wavy hair 2a",
                "2a wavy hair texture",
                "wavy hair 2a examples",
            ],
            "2b": [
                "2b wavy hair",
                "type 2b hair",
                "medium wavy hair 2b",
                "2b wavy hair pattern",
                "wavy hair 2b examples",
            ],
            "2c": [
                "2c wavy hair",
                "type 2c hair",
                "2c wavy curly hair",
                "2c hair texture",
                "wavy hair 2c examples",
            ],
            "3a": [
                "3a curly hair",
                "type 3a hair",
                "loose curly hair 3a",
                "3a curl pattern",
                "curly hair 3a examples",
            ],
            "3b": [
                "3b curly hair",
                "type 3b hair",
                "tight curly hair 3b",
                "3b curl pattern",
                "curly hair 3b examples",
            ],
            "3c": [
                "3c curly hair",
                "type 3c hair",
                "tight curly hair 3c",
                "3c curl pattern",
                "curly hair 3c examples",
            ],
            "4a": [
                "4a kinky coily hair",
                "type 4a hair",
                "4a coily hair texture",
                "4a hair pattern",
                "kinky coily hair 4a",
            ],
            "4b": [
                "4b kinky coily hair",
                "type 4b hair",
                "4b coily hair texture",
                "4b zigzag pattern",
                "kinky coily hair 4b",
            ],
            "4c": [
                "4c kinky coily hair",
                "type 4c hair",
                "4c coily hair texture",
                "4c hair pattern",
                "kinky coily hair 4c",
            ],
        }
        
    def search_images(self, query, page=0):
        """
        Search for images using SerpAPI
        
        Args:
            query: Search query
            page: Page number (usually just 0 or 1 work reliably)
            
        Returns:
            List of image URLs
        """
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.api_key,
            "ijn": page
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            image_urls = []
            if "images_results" in results:
                for image in results["images_results"]:
                    if "original" in image:
                        image_urls.append(image["original"])
                        
            return image_urls
        except Exception as e:
            print(f"  ⚠️  API Error: {e}")
            return []
    
    def download_image(self, url, save_path):
        """Download an image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return False
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except:
            return False
    
    def get_image_hash(self, filepath):
        """Generate hash of image to detect duplicates"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def collect_images_for_type(self, hair_type, target_count=4000):
        """
        Collect images for a specific hair type using multiple queries
        
        Args:
            hair_type: Hair type code (e.g., "1a", "2b")
            target_count: Target number of images to collect
        """
        if hair_type not in self.hair_type_queries:
            print(f"Error: '{hair_type}' is not a valid hair type!")
            return 0, 0
            
        queries = self.hair_type_queries[hair_type]
        type_dir = self.output_dir / hair_type
        type_dir.mkdir(exist_ok=True)
        
        # Count existing images
        existing_images = list(type_dir.glob("*.jpg")) + \
                         list(type_dir.glob("*.jpeg")) + \
                         list(type_dir.glob("*.png"))
        start_index = len(existing_images)
        
        print(f"\n{'='*60}")
        print(f"Collecting images for: {hair_type}")
        print(f"Existing images: {start_index}")
        print(f"Target: {target_count} NEW images")
        print(f"Search variations: {len(queries)}")
        print(f"{'='*60}\n")
        
        downloaded = 0
        api_calls = 0
        seen_hashes = set()
        seen_urls = set()  # Track URLs to avoid re-downloading
        
        # Try each query variation
        for query_idx, query in enumerate(queries):
            if downloaded >= target_count:
                break
                
            print(f"\n📝 Query {query_idx + 1}/{len(queries)}: '{query}'")
            print("─" * 60)
            
            # Try page 0 and 1 for each query (page 2+ rarely work)
            for page in [0, 1]:
                if downloaded >= target_count:
                    break
                
                print(f"  📄 Page {page}...", end=" ")
                image_urls = self.search_images(query, page=page)
                api_calls += 1
                
                if not image_urls:
                    print("No results")
                    continue
                
                # Filter out URLs we've already seen
                new_urls = [url for url in image_urls if url not in seen_urls]
                print(f"Found {len(image_urls)} images ({len(new_urls)} new)")
                
                for url in new_urls:
                    if downloaded >= target_count:
                        break
                    
                    seen_urls.add(url)
                    
                    # Create filename
                    file_index = start_index + downloaded
                    filename = f"{hair_type}_{file_index:05d}.jpg"
                    filepath = type_dir / filename
                    
                    # Download
                    if self.download_image(url, filepath):
                        try:
                            # Check for duplicates by hash
                            img_hash = self.get_image_hash(filepath)
                            if img_hash in seen_hashes:
                                filepath.unlink()
                                continue
                            seen_hashes.add(img_hash)
                            downloaded += 1
                            
                            if downloaded % 50 == 0:
                                print(f"    Progress: {downloaded}/{target_count} images")
                        except:
                            if filepath.exists():
                                filepath.unlink()
                    
                    time.sleep(0.3)  # Small delay
                
                time.sleep(1)  # Delay between pages
        
        print(f"\n✓ Completed {hair_type}: {downloaded} NEW images")
        print(f"  Total in folder: {start_index + downloaded}")
        print(f"  API calls used: {api_calls}")
        
        return downloaded, api_calls
    
    def collect_multiple_types(self, hair_types, images_per_type=4000):
        """
        Collect images for multiple hair types
        
        Args:
            hair_types: List of hair type codes
            images_per_type: Target images per type
        """
        total_downloaded = 0
        total_api_calls = 0
        
        print(f"\n{'='*60}")
        print(f"MULTI-QUERY DATA COLLECTION")
        print(f"{'='*60}")
        print(f"Hair types: {hair_types}")
        print(f"Target: {images_per_type} images per type")
        print(f"Strategy: Multiple search queries + pages 0-1")
        print(f"{'='*60}\n")
        
        for hair_type in hair_types:
            count, calls = self.collect_images_for_type(hair_type, images_per_type)
            total_downloaded += count
            total_api_calls += calls
            
            print(f"\n{'─'*60}")
            print(f"PROGRESS: {total_downloaded} total images, {total_api_calls} API calls")
            print(f"{'─'*60}\n")
            
            time.sleep(2)  # Delay between hair types
        
        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total images: {total_downloaded}")
        print(f"Total API calls: {total_api_calls}")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """
    IMPROVED VERSION - Uses multiple search queries
    """
    
    # ========================================
    # YOUR API KEY
    # ========================================
    API_KEY = "4af8a5dbf43754b78c324469ab6df3dff9abeaebac9eeb1eacf1f432baa6547f"
    
    # ========================================
    # YOUR COLLECTION PLAN
    # ========================================
    
    # Which hair types to collect
    my_hair_types = ["1a", "2a", "2b"]
    
    # How many images per type (we'll use multiple queries to get this many)
    images_per_type = 4000
    
    # ========================================
    # RUN COLLECTION
    # ========================================
    
    if API_KEY == "your_serpapi_key_here":
        print("\n❌ ERROR: Set your API key first!\n")
        return
    
    # Initialize scraper
    scraper = HairTypeImageScraperV2(
        api_key=API_KEY,
        output_dir="data/serpapi_raw"
    )
    
    # Collect images
    scraper.collect_multiple_types(
        hair_types=my_hair_types,
        images_per_type=images_per_type
    )
    
    print("\n✅ Done! Your images are in data/serpapi_raw/")
    print("\nNOTE: Google Images API has limitations on pagination.")
    print("This version uses multiple search queries to get more images.")
    print(f"Expected API calls: ~{len(my_hair_types) * 5 * 2} (5 queries × 2 pages × {len(my_hair_types)} types)\n")


if __name__ == "__main__":
    main()