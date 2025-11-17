# datacollection/yolo/yolo.py
# This script uses the YOLOv8 model to filter the SerpAPI dataset by only including those with one person
# Run this after collecting the raw images with the scraper. It will create a new folder with only the filtered images.

import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json

class PersonDetectorFilter:
    def __init__(self, dataset_dir="data/serpapi_raw", output_dir="data/yolo_filtered_serapi"):

        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load YOLOv8 model 
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt') 
        print("Model loaded successfully!\n")
        
        # processing stats to keep track of how many images were processed, how many had 0, 1, or multiple people, and any errors
        self.stats = {
            'total_processed': 0,
            'one_person': 0,
            'zero_people': 0,
            'multiple_people': 0,
            'errors': 0,
            'by_type': {}
        }
    
    def count_people_in_image(self, image_path):
        try:
            # Run YOLO detection
            results = self.model(image_path, verbose=False)
            
            # Count person detections (class 0 is 'person' in COCO dataset)
            person_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Class 0 is person, use confidence threshold
                    if class_id == 0 and confidence > 0.5:
                        person_count += 1
            
            return person_count
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def filter_hair_type_folder(self, hair_type):
        """
        Distinguish hair type subfolders
        
        Args:
            hair_type: Hair type code (e.g., "1", "2b")
        """
        source_dir = self.dataset_dir / hair_type
        target_dir = self.output_dir / hair_type
        target_dir.mkdir(exist_ok=True)
        
        if not source_dir.exists():
            print(f"Directory not found: {source_dir}")
            return
        
        # Get all image files
        image_files = list(source_dir.glob("*.jpg")) + \
                     list(source_dir.glob("*.jpeg")) + \
                     list(source_dir.glob("*.png"))
        
        print(f"\n{'='*60}")
        print(f"Processing: {hair_type}")
        print(f"Found {len(image_files)} images")
        print(f"{'='*60}\n")
        
        type_stats = {
            'total': len(image_files),
            'one_person': 0,
            'zero_people': 0,
            'multiple_people': 0,
            'errors': 0
        }
        
        kept_count = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Filtering {hair_type}"):
            self.stats['total_processed'] += 1
            
            person_count = self.count_people_in_image(str(img_path))
            
            if person_count is None:
                type_stats['errors'] += 1
                self.stats['errors'] += 1
            elif person_count == 1:
                # Keep this image - exactly one person
                target_path = target_dir / f"{hair_type}_{kept_count:05d}.jpg"
                shutil.copy2(img_path, target_path)
                kept_count += 1
                type_stats['one_person'] += 1
                self.stats['one_person'] += 1
            elif person_count == 0:
                type_stats['zero_people'] += 1
                self.stats['zero_people'] += 1
            else:  # person_count > 1
                type_stats['multiple_people'] += 1
                self.stats['multiple_people'] += 1
        
        self.stats['by_type'][hair_type] = type_stats
        
        print(f"\n{hair_type} Results:")
        print(f" - KEPT (1 person): {type_stats['one_person']}")
        print(f" - Removed (0 people): {type_stats['zero_people']}")
        print(f" - Removed (2+ people): {type_stats['multiple_people']}")
        print(f" - ERRORS: {type_stats['errors']}")
        print(f"Retention rate: {type_stats['one_person']/type_stats['total']*100:.1f}%\n")
    
    def filter_all(self):
        """
        Filter all hair type folders
        """
        # Get all subdirectories
        hair_types = [d.name for d in self.dataset_dir.iterdir() if d.is_dir()]
        hair_types.sort()
        
        print(f"Found {len(hair_types)} hair type folders: {hair_types}\n")
        
        for hair_type in hair_types:
            self.filter_hair_type_folder(hair_type)
        
        # Save statistics
        self.save_statistics()
        self.print_final_summary()
    
    def save_statistics(self):
        """Save filtering statistics to JSON file"""
        stats_file = self.output_dir / "filtering_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
    
    def print_final_summary(self):
        """Print final summary of filtering process"""
        print(f"\n{'='*60}")
        print("FILTERING COMPLETE - FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f" - Kept (1 person): {self.stats['one_person']}")
        print(f" - Removed (0 people):{self.stats['zero_people']}")
        print(f" - Removed (2+ people): {self.stats['multiple_people']}")
        print(f" - ERRORS: {self.stats['errors']}")
        print(f"\nOverall retention rate: {self.stats['one_person']/self.stats['total_processed']*100:.1f}%")
        print(f"\nFiltered dataset saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Per-type summary
        print("\nPer Hair Type Summary:")
        print(f"{'Type':<8} {'Total':<8} {'Kept':<8} {'Rate':<8}")
        print("-" * 40)
        for hair_type, stats in sorted(self.stats['by_type'].items()):
            rate = stats['one_person'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{hair_type:<8} {stats['total']:<8} {stats['one_person']:<8} {rate:>6.1f}%")


def main():
    filter_system = PersonDetectorFilter(
        dataset_dir="data/serpapi_raw",
        output_dir="data/yolo_filtered_serapi"
    )
    
    # Filter all images
    filter_system.filter_all()


if __name__ == "__main__":
    main()