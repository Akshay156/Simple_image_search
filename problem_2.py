import os
import json
import pickle
import matplotlib.pyplot as plt
from lib.faiss_handler import FaissManager
from lib.embedding_operations import FeatureExtractor
from lib.database_handler import PostgresManager
from sklearn.metrics.pairwise import cosine_similarity
from lib.helpers import preprocess_product_image, crop_image_from_bbox, natural_sort_key
from lib.object_detection import YOLOv8

from PIL import Image
import numpy as np
from rembg import remove 
import cv2

class ShelfImageProcessor:
    """
    Class for processing shelf images and detecting products.
    """

    def __init__(self, config_path, input_shelf_image_path):
        """
        Initialize the ShelfImageProcessor object.

        Args:
        - config_path (str): Path to the configuration file.
        - input_shelf_image_path (str): Path to the input shelf image.
        """
        self.config_path = config_path
        self.input_shelf_image_path = input_shelf_image_path
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
        self.feature_extractor = FeatureExtractor()
        self.load_product_embeddings()

    def load_product_embeddings(self):
        """
        Load or generate product embeddings.
        """
        regenerate_pickle = False
        pickle_file_path = "pickle/product_embedding_dict.pkl"
        if not os.path.exists(pickle_file_path) or regenerate_pickle:
            product_embedding_dict = {}
            product_image_folder_path = self.config["product_image_folder"]
            product_image_path_array = [os.path.join(product_image_folder_path, file) for file in sorted(os.listdir(product_image_folder_path), key=natural_sort_key) if file.endswith(('.jpg', '.jpeg', '.png'))]
            for product_image_path in product_image_path_array:
                product_image = cv2.imread(product_image_path)
                processed_image = preprocess_product_image(product_image)
                boxes, _, _ = self.detector.detect_objects(processed_image)
            
                if len(boxes) > 0:
                    image = crop_image_from_bbox(processed_image, boxes[0])
                else: 
                    image = cv2.resize(product_image, (500,500))
                    image = self.remove_background(image)
                
                product_name = os.path.basename(product_image_path).split(".")[0]
                product_embedding_dict[product_name] = self.feature_extractor.extract_embeddings(image)

            with open(pickle_file_path, "wb") as pickle_file:
                pickle.dump(product_embedding_dict, pickle_file)
        else:
            with open(pickle_file_path, "rb") as pickle_file:
                self.product_embedding_dict = pickle.load(pickle_file)

    def remove_background(self, image_np, threshold=None):
        """
        Remove the background from the input image using rembg library.
        """
        input_image = Image.fromarray(image_np)
        output_image = remove(input_image)
        output_image_np = np.array(output_image)
        return output_image_np

    def process_shelf_image(self):
        """
        Process the input shelf image to detect products.
        """
        shelf_image = cv2.imread(self.input_shelf_image_path)
        original_image = shelf_image.copy()
        boxes, _, _ = self.detector.detect_objects(shelf_image)
        height, width, _ = shelf_image.shape

        for box in boxes:

            x1, y1, x2, y2 = box
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            if bbox_width >= 5 and bbox_height >= 5:

                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                cropped_image = shelf_image[int(y1):int(y2), int(x1):int(x2)]
                embedding = self.feature_extractor.extract_embeddings(cropped_image)
                
                # display_image = cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.imshow("cropped shelf image", cropped_image)
                # cv2.waitKey(1)

                for product_name, product_embedding in self.product_embedding_dict.items():
                    similarity_score = cosine_similarity(embedding.reshape(1, -1), product_embedding.reshape(1, -1))
                    if similarity_score > self.config["similarity_threshold"]:
                        shelf_image_id = os.path.basename(self.input_shelf_image_path).split('db')[1].split('.')[0]
                        product_id = product_name.split('qr')[1]
                        with open("text/solution_2.txt", "a") as solution_file:
                            solution_file.write(f"{shelf_image_id}, {product_id}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}\n")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_shelf_image_path = "/mnt/Data/PD/InfiViz/Problem_1/Dataset/product_detection_from_packshots/shelf_images/db1027.jpg"
    processor = ShelfImageProcessor("config.json", input_shelf_image_path)
    processor.process_shelf_image()