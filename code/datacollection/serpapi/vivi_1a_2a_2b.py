"""
Hair Type Image Scraper - Multi-Query Version

Uses multiple search variations to collect more images
(bc pagination alone doesn't work well with Google Images API)

Feel free to modify the search queries for each hair type to get better results!!
"""


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
                "thick straight hair"
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
                "natural hair type 2b"
            ],
            "2c": [
                "root-starting defined waves",
                "voluminous wavy texture",
                "frizz-resistant styling pattern",
                "dense wave formation",
                "thick S-shaped strands",
                "coarse wavy structure",
                "abundant wave volume",
                "stubborn styling waves",
                "full-bodied undulation",
                "heavy wavy density",
                "robust S-curve pattern",
                "bulky wave texture",
                "pronounced frizz waves",
                "substantial wavy body",
                "thick undulating strands",
                "root-level wave start",
                "resistant wave styling",
                "full wavy dimension",
                "dense S-formation hair",
                "voluminous frizzy waves",
            ],
            "3a": [
                "loose loop curls",
                "quarter-sized ringlets",
                "wide-diameter spirals",
                "bouncy loose coils",
                "large curl circumference",
                "relaxed ringlet pattern",
                "big loopy texture",
                "broad spiral formation",
                "spacious curl structure",
                "gentle ringlet definition",
                "wide coil diameter",
                "loose spring pattern",
                "large curly loops",
                "open spiral texture",
                "broad ringlet formation",
                "expansive curl pattern",
                "loose bouncing coils",
                "quarter-width spirals",
                "wide curly dimension",
                "relaxed coil structure",
            ],
            "3b": [
                "penny-sized ringlets",
                "well-defined bouncy spirals",
                "marker-diameter coils",
                "tight springy ringlets",
                "medium curl circumference",
                "defined bouncing pattern",
                "compact spiral formation",
                "mid-sized coil texture",
                "structured ringlet definition",
                "moderate curl diameter",
                "bouncy spring structure",
                "clear spiral delineation",
                "penny-width curls",
                "defined coil formation",
                "medium ringlet texture",
                "structured bouncing spirals",
                "compact curl pattern",
                "marker-sized coils",
                "precise ringlet definition",
                "mid-diameter spring texture",
            ],
            "3c": [
                "pencil-width spirals",
                "tightly-packed coils",
                "dense corkscrew texture",
                "narrow curl circumference",
                "compact spiral formation",
                "small-diameter ringlets",
                "crowded coil pattern",
                "tight spring structure",
                "closely-wound spirals",
                "pencil-sized curls",
                "dense ringlet formation",
                "narrow coil diameter",
                "compressed spiral texture",
                "packed corkscrew pattern",
                "small curl structure",
                "tight coil definition",
                "compact ringlet density",
                "narrow spring formation",
                "crowded spiral texture",
                "pencil-diameter coils",
            ],
            "4a": [
                "densely-packed springy coils",
                "S-shaped defined texture",
                "tight springy formation",
                "compact coil definition",
                "dense S-pattern structure",
                "springy curl density",
                "closely-wound S-coils",
                "packed spring texture",
                "defined tight formation",
                "dense springy spirals",
                "compact S-shaped coils",
                "crowded spring pattern",
                "tight density structure",
                "S-curl definition texture",
                "packed springy formation",
                "dense coil arrangement",
                "tight S-pattern hair",
                "compressed spring coils",
                "compact defined texture",
                "densely springy structure",
            ],
            "4b": [
                "zigzag coil pattern",
                "Z-shaped texture formation",
                "angular coil structure",
                "dryness-prone tight pattern",
                "sharp-angled coils",
                "zigzag density texture",
                "geometric Z-formation hair",
                "angular spring pattern",
                "bent coil structure",
                "Z-pattern tight texture",
                "sharp zigzag coils",
                "angular dense formation",
                "geometric coil pattern",
                "Z-shaped springy texture",
                "bent tight structure",
                "zigzag arrangement hair",
                "angular coil density",
                "sharp Z-pattern texture",
                "geometric spring formation",
                "zigzag tight coils",
            ],
            "4c": [
                "extremely tight coils",
                "fragile dense texture",
                "minimal definition pattern",
                "very compressed coils",
                "delicate tight structure",
                "undefined dense formation",
                "ultra-tight coil texture",
                "fragile springy pattern",
                "extremely packed structure",
                "barely-defined coils",
                "ultra-dense formation",
                "very delicate texture",
                "tightly-compressed pattern",
                "minimal-definition coils",
                "fragile dense structure",
                "extremely tight formation",
                "ultra-packed texture",
                "very fragile coils",
                "compressed dense pattern",
                "ultra-tight delicate hair",
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
    API_KEY = "4af8a5dbf43754b78c324469ab6df3dff9abeaebac9eeb1eacf1f432baa6547f"
    
    ########################################
    # REPLACE HAIR TYPES that you are going to be collecting 
    ########################################

    # Which hair types to collect
    # Updated categories: "1" (straight), "2a", "2b", "2c" (wavy), "3a", "3b", "3c" (curly), "4a", "4b", "4c" (coily)
    my_hair_types = ["1", "2a", "2b"] 
    
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