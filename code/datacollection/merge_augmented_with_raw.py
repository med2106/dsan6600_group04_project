import os
import shutil

source_dir = "./data/yolo_filtered_serapi"
aug_dir = "./data/augmented"

for cls in os.listdir(aug_dir):
    aug_cls_path = os.path.join(aug_dir, cls)
    target_cls_path = os.path.join(source_dir, cls)
    
    # Skip if not a directory
    if not os.path.isdir(aug_cls_path):
        continue
    
    # Count files before
    original_count = len(os.listdir(target_cls_path))
    
    # Copy each augmented image to the original directory
    copied = 0
    for img_file in os.listdir(aug_cls_path):
        src = os.path.join(aug_cls_path, img_file)
        dst = os.path.join(target_cls_path, img_file)
        
        # Avoid overwriting existing files
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    
    new_count = len(os.listdir(target_cls_path))
    print(f"{cls}: {original_count} → {new_count} images (+{copied} augmented)")

print("\nDone! All augmented images merged.")