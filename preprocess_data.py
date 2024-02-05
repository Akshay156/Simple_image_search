import json
import os
import cv2
from lib.helpers import natural_sort_key
from lib.faiss_handler import FaissManager
from lib.embedding_operations import FeatureExtractor
from lib.object_detection import YOLOv8
from lib.entity_processor import EntityProcessor
from lib.database_handler import PostgresManager
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def initialize_managers(config):
    database_config = config["database"]
    database_manager = PostgresManager(**database_config)
    database_manager.connect_to_database()
    database_manager.create_table('shelf_images_data')
    database_manager.create_table('product_images_data')
    
    shelf_index_file = config["faiss_manager"]["shelf_index_file"]
    product_index_file = config["faiss_manager"]["product_index_file"]
    faiss_manager = FaissManager(shelf_index_file, product_index_file)
    faiss_manager.load_indexes()

    return database_manager, faiss_manager

def initialize_components(config):
    feature_extractor = FeatureExtractor()
    
    yolov8_config = config["yolov8"]
    yolov8_detector = YOLOv8(**yolov8_config)
    
    return feature_extractor, yolov8_detector

def get_image_paths(folder_path):
    return [os.path.join(folder_path, file) for file in sorted(os.listdir(folder_path), key=natural_sort_key) if file.endswith(('.jpg', '.jpeg', '.png'))]

def process_shelf_images(shelf_image_path_array, entity_processor, faiss_manager, database_manager):
    for image_path in shelf_image_path_array:
        shelf_image = cv2.imread(image_path)
        last_index_id = faiss_manager.shelf_index.ntotal
        df = entity_processor.process_shelf_image(shelf_image, os.path.basename(image_path), last_index_id)
        save_shelf_data(df, database_manager)
        update_faiss_index(df, faiss_manager, entity="shelf")  # Pass entity type as "shelf"

def update_faiss_index(df, faiss_manager, entity):
    embeddings_list = df['embedding'].tolist()
    embeddings_list = [np.frombuffer(embeddings_list[i], dtype=np.float32) for i in range(len(embeddings_list))]
    for embedding in embeddings_list:
        faiss_manager.append_and_get_index_ids(embedding.reshape(1, -1), entity)
    faiss_manager.save_indexes()

def save_shelf_data(df, database_manager):
    required_columns = ['filename', 'embedding', 'bbox', 'index_id']
    if all(col in df.columns for col in required_columns):
        data = df[required_columns].values.tolist()
        database_manager.insert_data('shelf_images_data', data)
    else:
        print("Error: DataFrame is missing one or more required columns.")
def process_product_images(product_image_path_array, entity_processor, faiss_manager, database_manager):
    for image_path in product_image_path_array:
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
        box = None  # Initialize box as None
        last_index_id = faiss_manager.product_index.ntotal
        df = entity_processor.process_product_image(image, image_name, box, last_index_id)
        if df is not None:  # Check if DataFrame is not None
            save_product_data(df, database_manager)
            update_faiss_index(df, faiss_manager, entity="product")  # Pass entity type as "product"

def save_product_data(df, database_manager):
    if df is not None:
        required_columns = ['filename', 'bbox', 'embedding', 'index_id']
        if all(col in df.columns for col in required_columns):
            data = df[required_columns].values.tolist()
            database_manager.insert_data('product_images_data', data)
        else:
            print("Error: DataFrame is missing one or more required columns.")
    else:
        print("Error: DataFrame is None.")


def update_config(config, shelf_image_path_array):
    config["last_file_processed"] = shelf_image_path_array[-1] if shelf_image_path_array else None
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Load config
    config = load_config('config.json')

    # Initialize managers
    database_manager, faiss_manager = initialize_managers(config)

    # Initialize components
    feature_extractor, yolov8_detector = initialize_components(config)

    product_image_folder_path = config["product_image_folder"]
    shelf_image_folder_path = config["shelf_image_folder"]

    # Get all image paths
    product_image_path_array = get_image_paths(product_image_folder_path)
    shelf_image_path_array = get_image_paths(shelf_image_folder_path)

    # Remove elements before the last processed file
    last_file_processed = config.get("last_file_processed")
    if last_file_processed:
        last_file_full_path = os.path.join(shelf_image_folder_path, last_file_processed)
        try:
            last_processed_index = shelf_image_path_array.index(last_file_full_path)
            shelf_image_path_array = shelf_image_path_array[last_processed_index + 1:]
        except ValueError:
            print(f"Error: '{last_file_full_path}' is not in shelf_image_path_array.")

    # Initialize EntityProcessor
    entity_processor = EntityProcessor(yolov8_detector, feature_extractor)

    # Initialize FaissManager
    shelf_index_file = config["faiss_manager"]["shelf_index_file"]
    product_index_file = config["faiss_manager"]["product_index_file"]
    faiss_manager = FaissManager(shelf_index_file, product_index_file)

    # Load Faiss indexes
    faiss_manager.load_indexes()

    # Process shelf images
    process_shelf_images(shelf_image_path_array, entity_processor, faiss_manager, database_manager)

    # Process product images
    process_product_images(product_image_path_array, entity_processor, faiss_manager, database_manager)

    # Update config
    update_config(config, shelf_image_path_array)

    # Close database connection
    database_manager.close_connection()

if __name__ == "__main__":
    main()
