import psycopg2
from psycopg2 import Error
import json

class PostgresManager:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cur = None
        self.config_path = 'config.json'

    def connect_to_database(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            print("Connected to database successfully!")
        except Error as e:
            print(f"Error connecting to database: {e}")

    def create_table(self, table_name):
        if self.cur is None:
            print("Database connection is not established.")
            return
        
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                filename VARCHAR(255),
                embedding BYTEA,
                bbox BYTEA,
                index_id INTEGER
            )
        """
        try:
            self.cur.execute(create_table_query)
            print(f"Table '{table_name}' created successfully.")
        except Exception as e:
            print(f"Error creating table: {e}")

    def insert_data(self, table_name, data):        
        insert_query = f"INSERT INTO {table_name} (filename, embedding, bbox, index_id) VALUES (%s, %s, %s, %s)"
        try:
            delete_duplicate = True
            for item in data:
                filename = item[0]
                embedding = item[1]
                bbox = item[2]
                index_id = item[3]

                # Check if entries with the same filename exist
                self.cur.execute(f"SELECT * FROM {table_name} WHERE filename = %s", (filename,))
                existing_entries = self.cur.fetchall()

                # If entries exist, delete them
                if len(existing_entries) > 0 and delete_duplicate:
                    self.cur.execute(f"DELETE FROM {table_name} WHERE filename = %s", (filename,))

                delete_duplicate = False
                
                # Insert new data
                self.cur.execute(insert_query, (filename, psycopg2.Binary(embedding), psycopg2.Binary(bbox), index_id))
            print("Data inserted successfully!")
        except Error as e:
            print(f"Error inserting data: {e}")

    def retrieve_data_by_index_id(self, table_name, index_id):
        if self.cur is None:
            print("Database connection is not established.")
            return None

        try:
            # Convert index_id to regular Python integer
            index_id_int = int(index_id)

            # Execute SQL query with index_id_int
            self.cur.execute(f"SELECT * FROM {table_name} WHERE index_id = %s", (index_id_int,))
            result = self.cur.fetchone()
            return result  # Assuming only one row matches the index_id
        except Error as e:
            print(f"Error retrieving data: {e}")

    def retrieve_data(self, table_name):
        if self.cur is None:
            print("Database connection is not established.")
            return None

        try:
            # Execute SQL query to fetch all rows
            self.cur.execute(f"SELECT * FROM {table_name}")
            results = self.cur.fetchall()
            return results
        except Error as e:
            print(f"Error retrieving data: {e}")

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("Connection closed successfully!")

    def update_last_processed_image(self, last_processed_image):
        try:
            with open(self.config_path, 'r') as config_file:
                config_data = json.load(config_file)

            config_data['last_processed_image'] = last_processed_image

            with open(self.config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)

            print("Config file updated successfully!")
        except Exception as e:
            print(f"Error updating config file: {e}")


    def retrieve_data_by_image_path(self, table_name, image_path):
        """
        Retrieve data from the database based on the provided image path and table name.

        Args:
        - table_name (str): The name of the table from which to retrieve the data.
        - image_path (str): The path of the image to match against in the database.

        Returns:
        A tuple containing the retrieved data if found, or None if no data is found.
        """
        if self.cur is None:
            print("Database connection is not established.")
            return None

        try:
            # Execute SQL query to fetch data based on image path
            self.cur.execute(f"SELECT * FROM {table_name} WHERE filename = %s", (image_path,))
            result = self.cur.fetchone()
            return result
        except Error as e:
            print(f"Error retrieving data: {e}")