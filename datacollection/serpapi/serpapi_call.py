# datacollection/serpapi/serpapi_call.py
# This script uses the SerpAPI to collect images for different hair types based on multiple search

# Methodology: 
# 1. For each hair type, query a list of search queries that are distinct with minimal word overlap (max 2 shared words)
    # Uses multiple search variations to collect more images
    # bc pagination alone doesn't work well with Google Images API

# 2. For each query, we search for images using the SerpAPI and download
# 3. We keep track of how many images we have downloaded for each hair type 
# 4. We also keep track of how many API calls we have made to stay within limits


### YOU MUST MODIFY: 
# 1. API_KEY: Your SerpAPI key (line 461)
# 2. my_hair_types: List of hair types you want to collect (line 469)
# both of these are in the main() function^ 

import os
import requests
from serpapi import GoogleSearch
import time
from pathlib import Path
import hashlib
from urllib.parse import urlparse

class HairTypeImageScraperV2:
    def __init__(self, api_key, output_dir="data/serpapi_raw"):
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
        # Each query is distinct with minimal word overlap (max 2 shared words)
        # Type 1 combines 1a, 1b, 1c characteristics
        self.hair_type_queries = {
        "1": [
            "type 1 straight hair",
            "1a straight hair",
            "1b straight hair",
            "1c straight hair",
            "straight glossy hair",
            "fine sleek strands",
            "pin straight texture hair",
            "silky flowing straight hair",
            "body volume straight",
            "coarse thick straight",
            "flat hair",
            "smooth shiny hair",
            "natural straight hairstyle",
            "straight hair style",
            "voluminous sleek texture",
            "curl resistant straight hair",
            "straight natural flow",
            "sleek natural straight hair",
            "thin straight hair",
            "thick straight hair",
            "type 1 hair care",
            "straight hair no bends",
            "naturally straight hair",
            "sleek straight hair products",
            "straight fine hair care"
        ],
        "2a": [
            "2a wavy hair",
            "loose wavy hair",
            "fine loose wavy hair",
            "type 2a hair care",
            "natural wavy hair 2a",
            "type 2a frizzy hair care",
            "hair type 2a style",
            "wavy beach hairstyle",
            "loose beachy waves",
            "fine wavy texture hair",
            "light subtle s-curve hair",
            "lightweight wavy strands",
            "frizz prone loose wavy hair",
            "loose wavy hair texture",
            "root-to-mid straight waves",
            "wavy hair less volume at roots",
            "wide s-shaped wavy hairstyle",
            "wide s shape hair",
            "type 2a hair s-shaped waves",
            "2a hair beach waves",
            "gentle wavy hairstyle",
            "soft cascading hair curls",
            "subtle wavy hair",
            "fine loose s-shaped wavy hair",
            "lightweight S-pattern hair"
        ],
        "2b": [
            "type 2b wavy hair",
            "medium wavy hair",
            "2b hair care",
            "defined wavy hair 2b",
            "type 2b hair chart classification",
            "natural 2b waves",
            "midlength waves 2b hair type",
            "s-shape curl 2b hair",
            "hairstyle for 2b wavy hair",
            "frizz-prone wavy hair 2b",
            "defined S-shaped waves",
            "frizz-prone wavy texture",
            "moderate wavy hair",
            "pronounced S-curve hair",
            "frizzy wavy hair texture",
            "medium thick waves 2b",
            "structured S-pattern hair",
            "emphasized wavy bends 2b",
            "medium-texture waves 2b type hair",
            "marked S-formation hair",
            "natural hair type 2b",
            "2b waves close to head",
            "2b wavy hair products",
            "defined waves 2b texture",
            "type 2b anti-frizz routine"
        ],
        "2c": [
            "type 2c wavy hair",
            "2c hair type",
            "thick wavy hair 2c",
            "2c hair care routine",
            "coarse wavy hair 2c",
            "type 2c frizzy hair",
            "2c wavy hair products",
            "defined waves 2c hair",
            "voluminous wavy hair 2c",
            "2c hair styling tips",
            "thick s-shaped waves",
            "2c hair curl definition",
            "dense wavy texture 2c",
            "2c wavy to curly hair",
            "hair type 2c texture",
            "frizz control 2c hair",
            "2c wave pattern hair",
            "natural 2c wavy hair",
            "2c hair root volume",
            "thick frizzy wavy hair 2c",
            "2c waves starting at roots",
            "2c hair with curls",
            "voluminous 2c wave texture",
            "2c mixed wave curl pattern",
            "women's type 2c hair"
        ],
        "3a": [
            "type 3a curly hair",
            "3a curls hair type",
            "loose curly hair 3a",
            "3a hair care routine",
            "big curls 3a hair",
            "type 3a curl pattern",
            "3a curly hair products",
            "loose spiral curls 3a",
            "3a hair styling",
            "defined 3a curls",
            "3a curly hair texture",
            "bouncy curls 3a",
            "natural 3a curly hair",
            "3a hair moisture routine",
            "large curl pattern 3a",
            "3a curly hairstyles",
            "type 3a ringlets",
            "3a curl definition",
            "loose coily hair 3a",
            "women's 3a curly hair",
            "3a large loop curls",
            "quarter-sized curls 3a",
            "3a sidewalk chalk curl size",
            "loose loopy curls 3a",
            "3a big spiral curls"
        ],
        "3b": [
            "type 3b curly hair",
            "3b curls hair type",
            "tight curly hair 3b",
            "3b hair care routine",
            "springy curls 3b hair",
            "type 3b curl pattern",
            "3b curly hair products",
            "defined spiral curls 3b",
            "3b hair styling tips",
            "bouncy 3b curls",
            "3b curly hair texture",
            "natural 3b curly hair",
            "3b hair shrinkage",
            "medium curl pattern 3b",
            "3b curly hairstyles",
            "type 3b ringlets",
            "3b curl definition products",
            "corkscrew curls 3b",
            "women's 3b curly hair",
            "3b hair frizz control",
            "3b penny-sized curls",
            "medium width ringlets 3b",
            "3b defined bouncy spirals",
            "3b medium curly texture",
            "springy 3b curl pattern"
        ],
        "3c": [
            "type 3c curly hair",
            "3c curls hair type",
            "tight curly hair 3c",
            "3c hair care routine",
            "dense curly hair 3c",
            "type 3c curl pattern",
            "3c curly hair products",
            "tight corkscrew curls 3c",
            "3c hair styling",
            "defined 3c curls",
            "3c curly hair texture",
            "natural 3c curly hair",
            "3c hair moisture",
            "small curl pattern 3c",
            "3c curly hairstyles",
            "type 3c coils",
            "3c curl definition",
            "tight spiral curls 3c",
            "women's 3c curly hair",
            "3c hair shrinkage issues",
            "3c pencil-width curls",
            "dense corkscrew texture 3c",
            "3c tight spiral pattern",
            "3c packed curls hair",
            "compact 3c curl texture"
        ],
        "4a": [
            "type 4a coily hair",
            "4a coils hair type",
            "kinky curly hair 4a",
            "4a hair care routine",
            "tight coily hair 4a",
            "type 4a curl pattern",
            "4a coily hair products",
            "s-pattern coils 4a",
            "4a hair styling",
            "defined 4a coils",
            "4a kinky hair texture",
            "natural 4a coily hair",
            "4a hair moisture routine",
            "tight s-curl pattern 4a",
            "4a coily hairstyles",
            "type 4a natural hair",
            "4a hair shrinkage",
            "springy coils 4a",
            "women's 4a kinky hair",
            "4a afro hair care",
            "4a soft springy coils",
            "densely packed 4a coils",
            "4a s-shaped texture",
            "4a tightly coiled hair",
            "springy 4a natural hair"
        ],
        "4b": [
            "type 4b coily hair",
            "4b coils hair type",
            "kinky hair 4b",
            "4b hair care routine",
            "tight kinky hair 4b",
            "type 4b curl pattern",
            "4b coily hair products",
            "z-pattern coils 4b",
            "4b hair styling tips",
            "4b natural hair texture",
            "4b kinky coily hair",
            "natural 4b coily hair",
            "4b hair moisture routine",
            "zigzag curl pattern 4b",
            "4b coily hairstyles",
            "type 4b natural hair",
            "4b hair shrinkage",
            "dense coily hair 4b",
            "women's 4b kinky hair",
            "4b afro textured hair",
            "4b z-shaped coils",
            "4b zigzag pattern hair",
            "coarse 4b coily texture",
            "4b less defined coils",
            "4b kinky hair breakage prevention"
        ],
        "4c": [
            "type 4c coily hair",
            "4c coils hair type",
            "kinky coily hair 4c",
            "4c hair care routine",
            "tight kinky hair 4c",
            "type 4c curl pattern",
            "4c coily hair products",
            "4c hair styling",
            "4c natural hair texture",
            "dense coily hair 4c",
            "4c kinky hair care",
            "natural 4c coily hair",
            "4c hair moisture routine",
            "tightly coiled hair 4c",
            "4c coily hairstyles",
            "type 4c natural hair",
            "4c hair extreme shrinkage",
            "fragile coily hair 4c",
            "women's 4c kinky hair",
            "4c afro hair texture",
            "4c no defined curl pattern",
            "4c extremely tight coils",
            "4c hair breakage prevention",
            "4c kinky hair moisture",
            "4c dense compact coils"
        ]
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
            print(f"API Error: {e}")
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
        
        print(f"\nCompleted {hair_type}: {downloaded} images")
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
        print(f"Hair types: {hair_types}")
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
        print(f"COLLECTION COMPLETE")
        print(f"Total images: {total_downloaded}")
        print(f"Total API calls: {total_api_calls}")
        print(f"Saved to: {self.output_dir}")


def main():
    ########################################
    # REPLACE WITH YOUR API KEY
    ########################################
    API_KEY = "your_serpapi_key_here"
    
    ########################################
    # REPLACE HAIR TYPES that you are going to be collecting 
    ########################################

    # Which hair types to collect
    # Updated categories: "1" (straight), "2a", "2b", "2c" (wavy), "3a", "3b", "3c" (curly), "4a", "4b", "4c" (coily)
    my_hair_types = ["1", "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4b", "4c"]  # Example: all types
    
    # How many images per type (we'll use multiple queries to get this many)
    images_per_type = 4000
    
    # ========================================
    # RUN COLLECTION
    # ========================================
    
    if API_KEY == "your_serpapi_key_here":
        print("\nERROR: Set your API key first!\n")
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
    
    print("\nDONE! Images saved in are in data/serpapi_raw/")
    print(f"Expected total API calls: ~{len(my_hair_types) * 40}\n")


if __name__ == "__main__":
    main()