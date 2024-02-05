![Preprocessing of dataset](images/pre_process.png?raw=true "Preprocessing of dataset")


![Solving Problem 1](images/problem_1.png?raw=true "Solving Problem 2")


![Solving Problem 2](images/problem_2.png?raw=true "Solving Problem 3")

Simple image based search

This repository contains the Retail Indexer System, a solution designed to streamline product search and detection processes within retail environments.

### System Components
- Object Detection: Utilizes pretrained YOLO model for object detection in images.
- Feature Extraction: Extracts embeddings from detected objects for similarity comparison. The model used here is pretrained ResNet50.
- Faiss Index Management: Manages Faiss indexes for fast similarity search.
- Database Integration: Integrates with PostgreSQL database for storing and retrieving image data.
- Shelf Processing: Processes shelf images to detect and locate products accurately.

### Configuration
Before running the system, ensure the `config.json` file is properly configured. Adjust parameters such as similarity threshold, model paths, and database credentials as needed.

Preprocessing

Download the models from the link.

Create the database and some folders:

bash

    mkdir models index_files text pickle
$ DB_NAME=$(jq -r '.database.dbname' config.json)
$ psql -U <username> -h <hostname> -c "CREATE DATABASE $DB_NAME"

Install requirements:

bash

    pip3 install -r requirements.txt

Run the program:

bash

    python3 preprocess_data.py

Basic Functionality:

    Get the data from the database.
    Create two tables in the database: shelf_image_data and product_image_data.
    Extract embeddings and save them in the database with bounding box and Faiss index as data.

This data or index can then be used in solving the problem.
Problem 1

After preprocessing, the system executes Problem 1, which involves searching for all products in shelf images and drawing bounding boxes around the detected products. This process utilizes object detection, feature extraction, and Faiss index management.
Problem Statement

Problem 1 requires identifying products within shelf images based on given product images. Each product image may appear in one or more shelf images, with multiple instances of the product possible in each shelf image. The task involves:

    Locating all instances of the product within shelf images.
    Marking the location of the product in the shelf image.
    Saving the relative data in solution.txt format.

Execution

To execute Problem 1, run the following command:

bash

    python3 problem_1.py

The program iterates over each product image, converting it into embeddings. These embeddings are then used to retrieve the nearest neighbors from the Faiss index. Cosine similarity is applied to filter out unwanted elements further. The results are saved in the solution_1.txt file inside the text folder.
Problem 2 Execution

Problem 2 is similar to Problem 1 but involves taking shelf images as input and finding all occurrences of products present in a given set of 100 products. Not all visible products in a shelf may be part of the given set of 100 products, and the system focuses only on these specified products.
Problem Statement

Problem 2 requires identifying occurrences of products within shelf images based on a given set of 100 products. The task involves:

    Identifying all occurrences of products from the specified set of product images found in shelf images.
    Finding all occurrences of products. Marking items in the specified set of shelf.
    Saving the relative data in solution.txt format.

Execution

First set the path of image inside the problem_2.py file

To execute Problem 2, run the following command:

bash

    python3 problem_2.py

The program iterates over each shelf image, detecting and extracting the products present. It then matches these products with the specified set of 100 products and saves the results in the solution_2.txt file inside the text folder.
Additional Information

For additional information and details on running the system, refer to the respective Python scripts where certain steps are commented. Also, for certain classes, there is in-depth description.



Sample `config.json`:
```json
{
    "similarity_threshold": 0.1,
    "database": {
        "dbname": "your_database_name",
        "user": "your_postgres_username",
        "password": "your_postgres_password"
    },
    "yolov8": {
        "model_path": "models/best.pt",
        "conf_thres": 0.45,
        "iou_thres": 0.5
    },
    "shelf_processor": {
        "conf_thres": 0.7,
        "iou_thres": 0.5,
        "min_bbox_width": 50,
        "min_bbox_height": 50
    },
    "faiss_manager": {
        "shelf_index_file": "index_files/shelf_images_index.idx",
        "product_index_file": "index_files/product_images_index.idx"
    },
    "shelf_image_folder": "directory_of_folder_with_shelf_images",
    "product_image_folder": "directory_of_folder_with_product_images"
}

