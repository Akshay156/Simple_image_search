import faiss
import numpy as np
import cv2
import pandas as pd

class FaissManager:
    def __init__(self, shelf_filename, product_filename):
        self.shelf_index = None
        self.product_index = None
        self.shelf_filename = shelf_filename
        self.product_filename = product_filename

    def create_indexes(self, shelf_embeddings_matrix, product_embeddings_matrix):
        self.shelf_index = faiss.IndexFlatL2(shelf_embeddings_matrix.shape[1])
        self.shelf_index.add(shelf_embeddings_matrix)

        self.product_index = faiss.IndexFlatL2(product_embeddings_matrix.shape[1])
        self.product_index.add(product_embeddings_matrix)

    def search_in_shelf(self, query_embedding, k=5):
        _, I = self.shelf_index.search(query_embedding.reshape(1, -1), k)
        return I[0]

    def search_in_product(self, query_embedding, k=5):
        _, I = self.product_index.search(query_embedding.reshape(1, -1), k)
        return I[0]

    def save_indexes(self):
        faiss.write_index(self.shelf_index, self.shelf_filename)
        faiss.write_index(self.product_index, self.product_filename)

    def load_indexes(self):
        try:
            self.shelf_index = faiss.read_index(self.shelf_filename)
            print(f"Loaded shelf index with {self.shelf_index.ntotal} entries.")
        except:
            self.shelf_index = faiss.IndexFlatL2(2048)  # Adjust the dimensionality as needed
            faiss.write_index(self.shelf_index, self.shelf_filename)
            print(f"Empty shelf Faiss index file created: {self.shelf_filename}")

        try:
            self.product_index = faiss.read_index(self.product_filename)
            print(f"Loaded product index with {self.product_index.ntotal} entries.")
        except:
            self.product_index = faiss.IndexFlatL2(2048)  # Adjust the dimensionality as needed
            faiss.write_index(self.product_index, self.product_filename)
            print(f"Empty product Faiss index file created: {self.product_filename}")

    def append_and_get_index_ids(self, embeddings_matrix, entity):
        """
        Append embeddings to the corresponding index based on the entity type (shelf or product).
        
        Args:
        - embeddings_matrix: The embeddings to be appended to the index.
        - entity: The type of entity (shelf or product).
        
        Returns:
        The total number of entries in the index after appending the embeddings.
        """
        if entity == "shelf":
            self.shelf_index.add(embeddings_matrix)
            faiss.write_index(self.shelf_index, self.shelf_filename)
            return self.shelf_index.ntotal
        elif entity == "product":
            self.product_index.add(embeddings_matrix)
            faiss.write_index(self.product_index, self.product_filename)
            return self.product_index.ntotal
