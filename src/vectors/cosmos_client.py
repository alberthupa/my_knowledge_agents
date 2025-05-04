import os
import json
import base64
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

'''
https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search
https://github.com/AzureCosmosDB/multi-agent-langgraph/blob/main/src/app/azure_cosmos_db.py
https://github.com/AzureCosmosDB/banking-multi-agent-workshop/blob/main/python/src/app/services/azure_cosmos_db.py
'''

load_dotenv()
# --- Configuration ---
# Replace with your actual connection string, database name, and container name
# It's recommended to use environment variables for sensitive information
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch" # Replace with your database name
CONTAINER_NAME = "knowledge-chunks" # Replace with your container name
PARTITION_KEY_PATH = "/id" # Replace with your partition key path (e.g., "/id")

# --- Connection ---
def get_cosmos_client(connection_string):
    """Parses connection string and returns CosmosClient."""
    parts = connection_string.split(';')
    uri = None
    key = None
    for part in parts:
        if part.startswith("AccountEndpoint="):
            uri = part.split("=")[1]
        elif part.startswith("AccountKey="):
            key_start_index = part.find("=") + 1
            key = part[key_start_index:]

    if not uri or not key:
        raise ValueError("Invalid connection string format")

    return CosmosClient(uri, credential=key)

def get_container_client(client, database_name, container_name):
    """Gets a reference to the database and container."""
    try:
        # First, try to get the database client
        database = client.get_database_client(database_name)
        print(f"Database '{database_name}' found.")

        # Then, try to get the container client
        container = database.get_container_client(container_name)
        print(f"Container '{container_name}' found.")

        print(f"Successfully connected to database '{database_name}' and container '{container_name}'.")
        return container
    except exceptions.CosmosResourceNotFoundError as e:
        # Check if the error is specifically for the database or container
        if f"dbs/{database_name}" in e.message:
             print(f"Error: Database '{database_name}' not found. Please ensure the database name is correct and exists.")
        elif f"colls/{container_name}" in e.message:
             print(f"Error: Container '{container_name}' not found in database '{database_name}'. Please ensure the container name is correct and exists.")
        else:
            print(f"Error: A Cosmos DB resource was not found: {e}")
        return None
    except exceptions.CosmosHttpResponseError as e:
        print(f"An unexpected Cosmos DB HTTP error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Data Upload ---
def upload_text_chunk(container, item_id, text_content, tags, vector_embedding=None, custom_properties=None):
    """Uploads a text chunk with tags and custom properties to the container."""
    document = {
        "id": item_id,
        "text": text_content,
        "tags": tags,
        "vector": vector_embedding # Placeholder for vector embedding
    }
    if custom_properties:
        document.update(custom_properties)
    try:
        container.upsert_item(document)
        print(f"Uploaded item with id: {item_id}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Error uploading item {item_id}: {e}")

# --- Vector Search ---
def vector_search(container, query_vector, top_k=5):
    """Performs a vector search against the container."""
    # NOTE: This is a basic example. The actual query might vary based on
    # the exact vector distance function and indexing policy configuration.
    # You will need to replace 'VectorDistance' with the correct function
    # and '/vector' with the path defined in your indexing policy if different.
    query = f"SELECT TOP 5 * FROM c ORDER BY VectorDistance(c.vector, {json.dumps(query_vector)})"
    
    results = []
    try:
        # Assuming query_items returns an iterable; adjust if using async client
        for item in container.query_items(query=query, enable_cross_partition_query=True):
             results.append(item)
        print(f"Found {len(results)} results for vector search.")
        return results[:top_k] # Return top_k results
    except exceptions.CosmosHttpResponseError as e:
        print(f"Error during vector search: {e}")
        return []

# --- Hybrid Search ---
def hybrid_search(container, keyword_query, query_vector, top_k=5):
    """Performs a hybrid search (keyword and vector) against the container."""
    # NOTE: This is a basic example. The actual query might vary based on
    # the exact vector distance function and indexing policy configuration.
    # You will need to replace 'VectorDistance' with the correct function
    # and '/vector' with the path defined in your indexing policy if different.
    # The keyword search is a simple LIKE query on the 'text' field.
    query = f"SELECT TOP 5 * FROM c WHERE CONTAINS(c.text, '{keyword_query}') ORDER BY VectorDistance(c.vector, {json.dumps(query_vector)})"

    results = []
    try:
        for item in container.query_items(query=query, enable_cross_partition_query=True):
             results.append(item)
        print(f"Found {len(results)} results for hybrid search.")
        return results[:top_k] # Return top_k results
    except exceptions.CosmosHttpResponseError as e:
        print(f"Error during hybrid search: {e}")
        return []


# --- Test Key Decoding ---
def test_key_decoding(connection_string):
    """Tests base64 decoding of the Account Key."""
    parts = connection_string.split(';')
    key = None
    for part in parts:
        if part.startswith("AccountKey="):
            key_start_index = part.find("=") + 1
            key = part[key_start_index:]
            break # Found the key, no need to continue

    if not key:
        print("AccountKey not found in connection string.")
        return

    print(f"Attempting to decode AccountKey: {key}")
    try:
        decoded_key = base64.b64decode(key)
        print("AccountKey decoding successful.")
        # print(f"Decoded key (first 10 bytes): {decoded_key[:10]}...") # Optional: print part of decoded key
    except Exception as e:
        print(f"AccountKey decoding failed: {e}")


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Test key decoding
    test_key_decoding(COSMOS_CONNECTION_STRING)

    # Replace with your actual database and container names
    db_name = "hupi-loch"
    container_name = "knowledge-chunks"

    cosmos_client = get_cosmos_client(COSMOS_CONNECTION_STRING)
    if cosmos_client:
        container_client = get_container_client(cosmos_client, db_name, container_name)

        if container_client:
            # Example of uploading a text chunk with custom properties
            custom_props = {
                "source_file": "example_document.txt",
                "processing_date": "2025-05-04",
                "category": "documentation"
            }
            upload_text_chunk(
                container_client,
                item_id="doc2",
                text_content="This is the first document about Azure Cosmos DB blah blah",
                tags=["azure", "cosmosdb"],
                custom_properties=custom_props
            )

            # Example of uploading a text chunk with a vector embedding
            # In a real application, you would generate this vector using an embedding model
            example_vector = [0.1] * 1024 # Replace with a vector of the correct dimension (e.g., 128 for many models)
            
            upload_text_chunk(
                container_client,
                item_id="doc4_with_vector",
                text_content="This document has a vector embedding.",
                tags=["vector", "example"],
                vector_embedding=example_vector
            )
            
            # Example of performing a vector search (replace with a real query vector)
            # You would typically generate this vector from your query text using an embedding model
            example_query_vector = [0.1] * 1024 # Replace with a vector of the correct dimension (e.g., 1024)
            search_results = vector_search(container_client, example_query_vector)
            print("\nVector Search Results:")
            for result in search_results:
                print(result)

            # Example of performing a hybrid search (keyword and vector)
            # Replace with a real keyword query and query vector
            example_keyword_query = "Azure Cosmos DB"
            example_hybrid_query_vector = [0.2] * 1024 # Replace with a vector of the correct dimension (e.g., 1024)
            #hybrid_search_results = hybrid_search(container_client, example_keyword_query, example_hybrid_query_vector)
            #print("\nHybrid Search Results:")
            #for result in hybrid_search_results:
            #    print(result)