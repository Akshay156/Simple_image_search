import cv2
import pandas as pd
from lib.helpers import preprocess_product_image, crop_image_from_bbox

class EntityProcessor:
    """
    EntityProcessor class for processing shelf and product images.

    @param detector: Object detection model for detecting objects in images.
    @param feature_extractor: Feature extraction model for extracting embeddings from images.
    """
    def __init__(self, detector, feature_extractor):
        self.detector = detector
        self.feature_extractor = feature_extractor

    def process_shelf_image(self, shelf_image, image_name, last_index_id):
        """
        Process shelf image to detect objects, extract embeddings, and create a DataFrame with the data.

        @param shelf_image: Input shelf image.
        @param image_name: Name of the image file.
        @param last_index_id: Last index ID in the Faiss index.
        @return: DataFrame containing image data (filename, bounding box, embedding, index_id).
        """
        # Detect Objects
        boxes, _, _ = self.detector.detect_objects(shelf_image)

        # Get image height and width
        height, width, _ = shelf_image.shape

        data = []

        # Iterate over all detected bounding boxes
        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box

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
                cropped_image = shelf_image[int(y1):int(y2), int(x1):int(x2)]

                # Get embeddings for the cropped image
                embeddings = self.feature_extractor.extract_embeddings(cropped_image)

                # Save image name, bounding box, and embedding for DataFrame
                data.append({
                    'filename': image_name,
                    'bbox': box,
                    'embedding': embeddings.tobytes(),  # Convert embeddings to bytes
                    'index_id': last_index_id
                })
                last_index_id += 1

                # Display cropped image with the file name

                # Draw rectangle on the shelf image
                cv2.rectangle(shelf_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display the shelf image with bounding boxes

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        return df
    
    def process_product_image(self, product_image, image_name, box, last_index_id):
        """
        Process product image to detect objects, extract embeddings, and create a DataFrame with the data.

        @param product_image: Input product image.
        @param image_name: Name of the image file.
        @param box: Bounding box coordinates of the product (not used for product images).
        @param last_index_id: Last index ID in the Faiss index.
        @return: DataFrame containing image data (filename, bounding box, embedding, index_id).
        """
        # Process product image and extract embeddings
        processed_image = preprocess_product_image(product_image)
        boxes, _, _ = self.detector.detect_objects(processed_image)
        
        if len(boxes) > 0:
            cropped_image = crop_image_from_bbox(processed_image, boxes[0])

            embeddings = self.feature_extractor.extract_embeddings(cropped_image)

            # Construct a dictionary with product image data
            data = {
                'filename': image_name,
                'bbox': boxes[0],  # Take the first detected bounding box
                'embedding': embeddings.tobytes(),  # Convert embeddings to bytes
                'index_id': last_index_id
            }

            # Create a DataFrame with the product image data
            df = pd.DataFrame([data])

            return df
        else:
            return None
