"""
Hair Type Image Scraper - Team Version
Allows each team member to collect specific hair types
"""

import os
import requests
from serpapi import GoogleSearch
import time
from pathlib import Path
import hashlib
from urllib.parse import urlparse

class HairTypeImageScraper:
    def __init__(self, api_key, output_dir="hair_type_dataset"):
        """
        Initialize the scraper
        
        Args:
            api_key: Your SerpAPI key
            output_dir: Directory to save images
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define ALL hair types and search queries
        self.all_hair_types = {
            "1a": "1a straight hair",
            "1b": "1b straight hair", 
            "1c": "1c straight hair",
            "2a": "2a wavy hair",
            "2b": "2b wavy hair",
            "2c": "2c wavy hair",
            "3a": "3a curly hair",
            "3b": "3b curly hair",
            "3c": "3c curly hair",
            "4a": "4a kinky coily hair",
            "4b": "4b kinky coily hair",
            "4c": "4c kinky coily hair"
        }
        
    def search_images(self, query, page=0):
        """
        Search for images using SerpAPI
        
        Args:
            query: Search query
            page: Page number (ijn parameter) - each page has ~100 images
            
        Returns:
            List of image URLs
        """
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.api_key,
            "ijn": page  # Google Images pagination parameter
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        image_urls = []
        if "images_results" in results:
            for image in results["images_results"]:
                if "original" in image:
                    image_urls.append(image["original"])
                    
        return image_urls
    
    def download_image(self, url, save_path):
        """
        Download an image from URL
        
        Args:
            url: Image URL
            save_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check if it's an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return False
            
            # Save the image
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def get_image_hash(self, filepath):
        """Generate hash of image to detect duplicates"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def collect_images_for_type(self, hair_type, target_count=3500, start_page=0):
        """
        Collect images for a specific hair type
        
        Args:
            hair_type: Hair type code (e.g., "1a", "2b")
            target_count: Target number of images to collect
            start_page: Starting page number (for continuing collection)
        """
        if hair_type not in self.all_hair_types:
            print(f"Error: '{hair_type}' is not a valid hair type!")
            print(f"Valid types: {list(self.all_hair_types.keys())}")
            return 0, 0
            
        query = self.all_hair_types[hair_type]
        type_dir = self.output_dir / hair_type
        type_dir.mkdir(exist_ok=True)
        
        # Count existing images to continue numbering
        existing_images = list(type_dir.glob("*.jpg")) + \
                         list(type_dir.glob("*.jpeg")) + \
                         list(type_dir.glob("*.png"))
        start_index = len(existing_images)
        
        print(f"\n{'='*60}")
        print(f"Collecting images for: {hair_type} - '{query}'")
        print(f"Existing images: {start_index}")
        print(f"Target: {target_count} NEW images")
        print(f"Starting from page: {start_page}")
        print(f"{'='*60}\n")
        
        downloaded = 0
        page = start_page
        seen_hashes = set()
        max_pages = start_page + (target_count // 100) + 2  # Calculate pages needed
        
        # Keep searching until we hit target or exhaust results
        while downloaded < target_count and page < max_pages:
            print(f"\n{'─'*60}")
            print(f"📄 Fetching page {page} (API call #{page - start_page + 1})")
            print(f"   Pagination: ijn={page} → Expected images {page*100+1}-{(page+1)*100}")
            print(f"{'─'*60}")
            
            image_urls = self.search_images(query, page=page)
            
            if not image_urls:
                print("⚠️  No more results found - reached end of available images")
                break
            
            print(f"✓ Found {len(image_urls)} new image URLs from page {page}")
            
            for idx, url in enumerate(image_urls):
                if downloaded >= target_count:
                    break
                
                # Create filename with continuing index
                file_index = start_index + downloaded
                filename = f"{hair_type}_{file_index:05d}.jpg"
                filepath = type_dir / filename
                
                # Download image
                print(f"  [{downloaded+1}/{target_count}] Downloading: {filename}...", end=" ")
                
                if self.download_image(url, filepath):
                    # Check for duplicates
                    try:
                        img_hash = self.get_image_hash(filepath)
                        if img_hash in seen_hashes:
                            print("DUPLICATE - removed")
                            filepath.unlink()
                            continue
                        seen_hashes.add(img_hash)
                        print("SUCCESS")
                        downloaded += 1
                    except Exception as e:
                        print(f"ERROR: {e}")
                        if filepath.exists():
                            filepath.unlink()
                else:
                    print("FAILED")
                
                # Small delay to be respectful
                time.sleep(0.5)
            
            page += 1
            time.sleep(2)  # Delay between API calls
        
        print(f"\n✓ Completed {hair_type}: {downloaded} NEW images downloaded")
        print(f"  Total images in folder: {start_index + downloaded}")
        print(f"  API calls used: {page - start_page}")
        return downloaded, page - start_page
    
    def collect_with_custom_ranges(self, collection_plan):
        """
        Collect images with custom page ranges for each hair type
        
        Args:
            collection_plan: List of tuples (hair_type, start_page, num_pages)
                           e.g., [("1a", 1, 40), ("2a", 0, 40), ("2b", 0, 20)]
        """
        total_downloaded = 0
        total_api_calls = 0
        
        print(f"\n{'='*60}")
        print(f"STARTING DATA COLLECTION - CUSTOM RANGES")
        print(f"{'='*60}")
        print(f"Collection plan:")
        for hair_type, start_page, num_pages in collection_plan:
            print(f"  {hair_type}: pages {start_page}-{start_page + num_pages - 1} ({num_pages} API calls)")
        print(f"Total estimated API calls: {sum(p[2] for p in collection_plan)}")
        print(f"{'='*60}\n")
        
        for hair_type, start_page, num_pages in collection_plan:
            target_images = num_pages * 100  # ~100 images per page
            count, api_calls = self.collect_images_for_type(
                hair_type, 
                target_count=target_images,
                start_page=start_page
            )
            total_downloaded += count
            total_api_calls += api_calls
            
            print(f"\n{'─'*60}")
            print(f"Progress Summary:")
            print(f"  Images collected so far: {total_downloaded}")
            print(f"  API calls used so far: {total_api_calls}")
            print(f"{'─'*60}\n")
        
        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total images downloaded: {total_downloaded}")
        print(f"Total API calls used: {total_api_calls}")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """
    SHARED GITHUB REPO CONFIGURATION
    
    Team members working in shared repo:
    1. Each person uses their own SerpAPI key
    2. Each person specifies their collection ranges below
    3. Everyone outputs to the SAME directory (data/serpapi_raw/)
    4. Files automatically continue numbering - no merge needed!
    """
    
    # ========================================
    # STEP 1: Add YOUR SerpAPI key here
    # ========================================
    API_KEY = "your_serpapi_key_here"
    
    # ========================================
    # STEP 2: Define YOUR collection plan
    # ========================================
    
    # Format: (hair_type, start_page, num_pages)
    # Example: ("1a", 1, 40) means collect pages 1-40 for hair type 1a
    
    # YOUR COLLECTION (Person 1):
    # - 1a: pages 1-40 (40 calls) - skipping page 0 since already collected
    # - 2a: pages 0-39 (40 calls) 
    # - 2b: pages 0-19 (20 calls)
    # Total: 100 API calls
    
    my_collection_plan = [
        ("1a", 1, 40),   # Pages 1-40 (skipping page 0)
        ("2a", 0, 40),   # Pages 0-39
        ("2b", 0, 20),   # Pages 0-19
    ]
    
    # ========================================
    # TEAMMATE'S COLLECTION PLAN (for reference)
    # ========================================
    # When your teammate runs this, they would use:
    # 
    # teammate_collection_plan = [
    #     ("2b", 20, 20),  # Pages 20-39 (continuing where you left off)
    #     ("2c", 0, 40),   # Pages 0-39
    #     ("3a", 0, 40),   # Pages 0-39
    # ]
    # 
    # They just need to:
    # 1. Change API_KEY to their key
    # 2. Uncomment their plan and comment out yours
    # 3. Run the script!
    
    # ========================================
    # STEP 3: Run the script!
    # ========================================
    
    if API_KEY == "your_serpapi_key_here":
        print("\n" + "="*60)
        print(" ❌ ERROR: API KEY NOT SET ".center(60))
        print("="*60)
        print("\nPlease replace 'your_serpapi_key_here' with your actual SerpAPI key")
        print("Example:")
        print('  API_KEY = "4af8a5dbf43754b78c324469ab6df3dff9abeaebac9eeb1eacf1f432ba"')
        print("\n" + "="*60 + "\n")
        return
    
    # Initialize scraper with shared output directory
    scraper = HairTypeImageScraper(
        api_key=API_KEY,
        output_dir="data/serpapi_raw"  # Shared directory!
    )
    
    # Collect images based on your plan
    scraper.collect_with_custom_ranges(my_collection_plan)
    
    print("\n✅ Done! Your images are in data/serpapi_raw/")
    print("📁 Commit and push to GitHub when ready!")
    print("📁 Your teammate can then pull and run their collection.\n")


if __name__ == "__main__":
    main()