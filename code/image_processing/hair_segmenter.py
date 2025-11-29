# NOTE: MediaPipe requires numpy<2 which is incompatible with the rest of the project env.
#       Recommended to run this script in a different environment with the right dependencies

from pathlib import Path
from tqdm import tqdm
import cv2
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


## PATHS (run this script from directory root or change path below)
model_path  = './code/image_processing/hair_segmenter.tflite'
input_dir1  = './data/yolo_filtered_serapi/'
input_dir2  = './data/augmented/'
output_dir  = './data/segmented/'

# sub-directories by hair type
hair_types = list(set(
    [sub_dir.name for sub_dir in Path(input_dir1).iterdir() if sub_dir.is_dir()] +
    [sub_dir.name for sub_dir in Path(input_dir2).iterdir() if sub_dir.is_dir()]
))
hair_types.sort()

Path(output_dir).mkdir(exist_ok=True)
for type in hair_types:
    Path(output_dir, type).mkdir(exist_ok=True)



## SEGMENTER MODEL CONFIG
options = vision.ImageSegmenterOptions(
    base_options = base_options.BaseOptions(
        model_asset_path = model_path,
        delegate = base_options.BaseOptions.Delegate.CPU # or .GPU (only on Ubuntu platforms)
    ),
    running_mode = vision.RunningMode.IMAGE,
    output_category_mask = False
)


## USER PARAMETERS

# set a threshold value for the segmenter's mask output
# to focus purely on regions with hair and minimize bounding box
mask_threshold = 0.7

# final image size based on CNN model
final_size = (600, 600)


## UTILITIES
def mask_background(image_path, confidence_mask, bg_color=(255, 255, 255)):
    """
    image_path      : original image path
    confidence_mask : 2D numpy array of binary [0, 1]
    bg_color        : RGB or BGR background color to apply on background
    """

    h, w = confidence_mask.shape
    image_bgr = cv2.imread(image_path)
    image_bgr = cv2.resize(image_bgr, (w, h)) # should be the same as original, but for thoroughness

    # Expand mask to 3 channels
    mask_3d = np.stack([confidence_mask]*3, axis=-1)

    bg = np.zeros_like(image_bgr) + np.array(bg_color, dtype=np.uint8)

    hair_region     = (image_bgr * mask_3d).astype(np.uint8)
    nonhair_region  = (bg * (1 - mask_3d)).astype(np.uint8)

    final_img = hair_region + nonhair_region
    return final_img


def get_bounding_box(mask, padding_ratio=0.05):
    """
    mask            : 2D numpy array of binary [0, 1]
    padding_ratio   : adds padding, default 5%
    """

    total_H, total_W = mask.shape
    ys, xs = np.where(mask)

    # test for empty image
    if len(xs) == 0:
        return None

    # find bounding co-ordinates
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min
    h = y_max - y_min

    if min(w, h) <= 0:
        return None

    # Add 5% padding
    pad = int(min(w, h) * padding_ratio)
    x1 = max(0, x_min - pad)
    y1 = max(0, y_min - pad)
    x2 = min(total_W, x_max + pad)
    y2 = min(total_H, y_max + pad)

    return (x1, y1, x2, y2)


def crop_and_resize(image, bbox):
    """
    image   : masked image to be cropped
    bbox    : bounding box co-ordinates
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, final_size) # interpolation=cv2.INTER_AREA
    return resized


## MAIN
def segment_images(segmenter, input_dir, hair_type):
    # Logger for current hair_type
    log_file = f'./code/image_processing/logs/logger_{hair_type}.log'
    logger = logging.getLogger(f'segmenter_{hair_type}')
    logger.setLevel(logging.WARNING)

    if not logger.handlers:
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    input_path = Path(input_dir, hair_type)
    image_list = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))

    for img in tqdm(image_list, desc=f"Segmenting images at {input_path} :"):
        try:
            image_path = str(Path(img))

            # Load image as MediaPipe image file
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logger.warning(f"SKIPPED   : {image_path} : OpenCV failed to read the image, file might be corrupted")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert BGR to RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform image segmentation and obtain confidence mask
            segmented_masks = segmenter.segment(mp_image)

            # index 1 for hair, shape as (H, W), probability values in [0,1]
            mask = segmented_masks.confidence_masks[1].numpy_view().copy()
            
            # filter out pixels with probabilities below the set threshold completely
            mask[mask < mask_threshold]    = 0
            mask[mask >= mask_threshold]   = 1

            # Mask background (non-hair regions) with a solid color
            masked_image = mask_background(image_path, mask)

            # Get bounding box of hair region
            bbox = get_bounding_box(mask)
            if bbox is None:
                logger.warning(f"SKIPPED   : {image_path} : No hair detected")
                continue

            # Crop image to bounding box and resize
            output_image = crop_and_resize(masked_image, bbox)

            # Save final image
            cv2.imwrite(str(Path(output_dir, hair_type, img.name)), output_image)

        except Exception as e:
            logger.error(f"ERROR     : {image_path}\n{e}", exc_info=True)
            continue


if __name__ == "__main__":
    try:
        ## Image Segmenter model instance
        with vision.ImageSegmenter.create_from_options(options) as segmenter:

            for input_dir in [input_dir1, input_dir2]:
                # Loop through hair types directories
                for hair_type in hair_types:
                    if not Path(input_dir, hair_type).is_dir():
                        continue

                    # loops through all images in given directory
                    segment_images(segmenter, input_dir, hair_type)
    except Exception as e:
        print('\n\n', e)