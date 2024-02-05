import faiss
import numpy as np
import cv2
import pandas as pd

class DataFrameManager:
    def __init__(self):
        pass

    def combine_dataframes(self, df_list):
        return pd.concat(df_list, ignore_index=True)

    def add_index_ids(self, df):
        df['faiss_index_id'] = np.arange(len(df))
        return df

    def get_respective_data(self, df, faiss_index_id):
        """
        Retrieve the respective data from the DataFrame based on the Faiss search result index.
        """
        matching_row = df[df['faiss_index_id'] == faiss_index_id]
        matching_image_data = matching_row['image_data'].values[0]  # Adjust column name accordingly
        return matching_image_data

class ExcelManager:
    def __init__(self):
        pass

    def save_to_excel(self, df, filename):
        df.to_excel(filename, index=False)

    def read_from_excel(self, filename):
        return pd.read_excel(filename)

import os
def get_respective_data(database_reply, shelf_image_folder_path):
    """
    Retrieve the respective data from the DataFrame based on the Faiss search result index.
    """
    try:
        filename, embedding, bbox, _ = database_reply  # Unpack the values from the database reply

        # Convert bytes to numpy arrays
        embedding = np.frombuffer(embedding, dtype=np.float32)  # Assuming the embedding are float32
        bbox = np.frombuffer(bbox, dtype=np.float32)  # Assuming the bounding box coordinates are float32

        # Combine shelf_image_folder_path with filename to get the full path
        image_path = os.path.join(shelf_image_folder_path, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Draw the bounding box on the matching image
        bbox = [int(x) for x in bbox]
        # x1, y1, x2, y2 = bbox
        # cv2.rectangle(matching_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


        return image, bbox, embedding  # Return the matching image with the bounding box drawn
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
