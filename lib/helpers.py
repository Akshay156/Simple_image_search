import cv2
from PIL import Image
import numpy as np
from rembg import remove 
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def crop_image_from_bbox(image, bbox):
        
        height, width, _ = image.shape
        
        # Get the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # Calculate bounding box width and height
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Filter out bounding boxes that are too small
        if bbox_width >= 5 and bbox_height >= 5:
            # Clip bounding box coordinates to image dimensions
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Crop the image using the bounding box coordinates
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            return cropped_image

        else: return None    

def preprocess_product_image(image):
    target_size = 300
    height, width, _ = image.shape
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    image = cv2.resize(image, (new_width, new_height))

    # Pad the image to size 450x450
    target_size = 640
    top = (target_size - new_height) // 2
    bottom = target_size - new_height - top
    left = (target_size - new_width) // 2
    right = target_size - new_width - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image

def resize_with_padding(image, target_width, target_height):
    """
    Resize the input image to the target dimensions while maintaining aspect ratio and adding padding.
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > target_width / target_height:
        # Image is wider than the target aspect ratio, resize based on width
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than the target aspect ratio, resize based on height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Add padding to match the target dimensions
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad

    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def is_object_visible(image_np, threshold=True, black_threshold=0.95):
    """
    Check if the object in the image is sufficiently visible.
    """
    if threshold is None:
        return True  # Default behavior
    else:
        # Calculate the ratio of object area to total image area
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        object_area = cv2.countNonZero(binary_mask)
        total_area = image_np.shape[0] * image_np.shape[1]
        object_percentage = object_area / total_area

        # Calculate the ratio of black area to total image area
        black_area = total_area - object_area
        black_percentage = black_area / total_area

        # Check if object percentage exceeds the threshold
        if object_percentage >= threshold and black_percentage < black_threshold:
            return True
        else:
            return False

