from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

datagen = ImageDataGenerator(
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

source_dir = "data/yolo_filtered_serapi"
target_count = 1200
aug_dir = "data/augmented"

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    # make sure that it is only looking at the directories
    if not os.path.isdir(cls_path):
        continue

    files = os.listdir(cls_path)

    if len(files) < target_count:
        needed = target_count - len(files)
        save_dir = os.path.join(aug_dir, cls)
        print(save_dir)
        # make directory if it doesnt exist
        os.makedirs(save_dir, exist_ok=True)

        # apply the generator and save with the class prefix
        generator = datagen.flow_from_directory(
            source_dir,
            classes=[cls],
            class_mode=None,
            save_to_dir=save_dir,
            save_prefix=cls,
            save_format="jpg",
            batch_size=1
        )

        for _ in range(needed):
            next(generator)
