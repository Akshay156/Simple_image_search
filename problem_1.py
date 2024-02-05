import os
import json
import matplotlib.pyplot as plt
from lib.faiss_handler import FaissManager
from lib.embedding_operations import FeatureExtractor
from lib.database_handler import PostgresManager
from sklearn.metrics.pairwise import cosine_similarity
from lib.helpers import preprocess_product_image, crop_image_from_bbox
from lib.object_detection import YOLOv8

from PIL import Image
import numpy as np
from rembg import remove 
import cv2

class ShelfImageSearch:
    """
    Class for searching shelf images for specific products.
    """

    def __init__(self, config_path):
        """
        Initialize the ShelfImageSearch object.

        Args:
        - config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.load_config()
        self.load_models()

    def load_config(self):
        """
        Load the configuration from the JSON file.
        """
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
    def load_models(self):
        """
        Load necessary models and configurations.
        """
        yolov8_config = self.config["yolov8"]
        self.detector = YOLOv8(**yolov8_config)
        shelf_index_file = self.config["faiss_manager"]["shelf_index_file"]
        product_index_file = self.config["faiss_manager"]["product_index_file"]
        db_config = self.config["database"]
        self.faiss_manager = FaissManager(shelf_index_file, product_index_file)
        self.faiss_manager.load_indexes()
        self.postgres_manager = PostgresManager(**db_config)
        self.postgres_manager.connect_to_database()
        self.feature_extractor = FeatureExtractor()

    def remove_background(self, image_np, threshold=None):
        """
        Remove the background from the input image using rembg library.
        """
        input_image = Image.fromarray(image_np)
        output_image = remove(input_image)
        output_image_np = np.array(output_image)
        return output_image_np

    def get_respective_data(self, image_path, bbox):
        """
        Retrieve the respective data from the dataframe based on the Faiss index.
        """
        matching_image = cv2.imread(image_path)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(matching_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return matching_image

    def search_for_item(self, image_path):
        """
        Search for an item using the given image path.

        Args:
        - image_path (str): Path to the query image.

        Returns:
        - matching_images_info (list): List of dictionaries containing information about matching images.
        - matching_images (list): List of matching images.
        - query_image (numpy.ndarray): Query image.
        """
        image = cv2.imread(image_path)
        processed_image = preprocess_product_image(image)
        boxes, _, _ = self.detector.detect_objects(processed_image)
        
        if len(boxes) > 0:
            image = crop_image_from_bbox(processed_image, boxes[0])
        else: 
            image = cv2.resize(image, (500,500))
            image = self.remove_background(image)
        
        query_embedding = self.feature_extractor.extract_embeddings(image)
        results = self.faiss_manager.search_in_shelf(query_embedding, k=150)

        matching_images_info = []
        matching_images = []

        similarity_threshold = self.config["similarity_threshold"]
        if results is not None:
            for result in results:
                index_id = result
                table_name = "shelf_images_data"
                data = self.postgres_manager.retrieve_data_by_index_id(table_name, index_id)
                if data is not None:
                    image_name, embedding_bytes, bbox_bytes, _ = data
                    matched_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    bbox = np.frombuffer(bbox_bytes, dtype=np.float32)
                    matched_image_path = os.path.join(self.config["shelf_image_folder"], image_name)
                    matching_image = self.get_respective_data(matched_image_path, bbox)
                    product_id = os.path.splitext(os.path.basename(image_path))[0].replace("qr", "")
                    shelf_id = os.path.splitext(os.path.basename(matched_image_path))[0].replace("db", "")
                    similarity_score = cosine_similarity(query_embedding.reshape(1, -1), matched_embedding.reshape(1, -1))

                    if similarity_score[0][0] > similarity_threshold:
                        matching_images_info.append({
                            "product_id": product_id,
                            "shelf_id": shelf_id,
                            "bbox": [int(coord) for coord in bbox],
                            "similarity_score": similarity_score[0][0]
                        })
                        matching_images.append(matching_image)

        else:
            print(f"No matching image found for: {image_path}")
        
        return matching_images_info, matching_images, image

    def run_search(self):
        """
        Perform the search for all images in the product image folder.
        """
        product_image_folder = self.config["product_image_folder"]
        for filename in os.listdir(product_image_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                _image_path = os.path.join(product_image_folder, filename)
                matching_images_info, matching_images, query_image = self.search_for_item(_image_path)
                if len(matching_images_info) > 0:
                    with open("text/solution_1.txt.txt", "a") as f:
                        for info in matching_images_info:
                            f.write(f"{info['product_id']}, {info['shelf_id']}, {','.join(map(str, info['bbox']))}\n")
                    for matching_image in matching_images:
                        cv2.imshow("Matching Image", matching_image)
                        cv2.imshow("Query Image", query_image)
                        cv2.waitKey(1)
            else:
                continue

if __name__ == "__main__":
    shelf_search = ShelfImageSearch("config.json")
    shelf_search.run_search()
