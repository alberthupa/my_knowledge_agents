import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv(override=True)


class SimpleCosmosClient:
    """
    A simple client for interacting with Azure Cosmos DB.
    """

    def __init__(
        self,
        connection_string: str,
        database_name: str,
        container_name: str,
        partition_key_path: str,
        vector_embedding_policy: dict,
        indexing_policy: dict,
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.container_name = container_name
        self.partition_key_path = partition_key_path
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_client = None
        self.database_client = None
        self.container_client = None

    def connect(self):
        """
        Connects to the Cosmos DB account and gets the database client.
        """
        try:
            # Parse connection string (inspired by cosmos_sketches.py)
            parts = self.connection_string.split(";")
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

            self.cosmos_client = CosmosClient(uri, key)
            print("CosmosClient initialized successfully.")

            # Get database client
            self.database_client = self.cosmos_client.get_database_client(
                self.database_name
            )
            print(f"Database '{self.database_name}' client obtained.")

            # get container client
            self.container_client = self.database_client.get_container_client(
                self.container_name
            )
            print(f"container '{self.container_name}' client obtained.")

        except exceptions.CosmosResourceNotFoundError:
            print(
                f"Error: Database '{self.database_name}' not found. Please ensure the database name is correct and exists."
            )
            self.database_client = None
            self.container_client = None
        except ValueError as e:
            print(f"Connection string error: {e}")
            self.cosmos_client = None
            self.database_client = None
            self.container_client = None
        except Exception as e:
            print(f"An unexpected error occurred during connection: {e}")
            self.cosmos_client = None
            self.database_client = None
            self.container_client = None

    def create_knowledge_chunks_container(self):
        """
        Creates the 'knowledge_chunks' container with specified policies if it doesn't exist.
        client.create_knowledge_chunks_container()
        """
        if not self.database_client:
            print("Cannot create container: Not connected to database.")
            return

        try:
            container = self.database_client.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path=self.partition_key_path),
                vector_embedding_policy=self.vector_embedding_policy,
                indexing_policy=self.indexing_policy,
                full_text_policy=full_text_policy,
            )
            print(f"Container '{self.container_name}' created or already exists.")
            return container
        except exceptions.CosmosResourceExistsError:
            print(f"Container '{self.container_name}' already exists.")
            return self.database_client.get_container_client(self.container_name)
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error creating container '{self.container_name}': {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during container creation: {e}")
            return None

    def delete_container(self, container_name: str):
        """
        Deletes the specified container.
        client.delete_container(CONTAINER_NAME)
        """
        if not self.database_client:
            print("Cannot delete container: Not connected to database.")
            return

        try:
            self.database_client.delete_container(container_name)
            print(f"Container '{container_name}' deleted successfully.")
        except exceptions.CosmosResourceNotFoundError:
            print(f"Container '{container_name}' not found.")
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error deleting container '{container_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred during container deletion: {e}")


# Configuration variables (as provided by the user)
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
CONTAINER_NAME = "knowledge-chunks"
# CONTAINER_NAME = "test_container"
PARTITION_KEY_PATH = "/id"

# Vector embedding policy (as provided by the user)
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "dimensions": 1024,
            "distanceFunction": "cosine",
        }
    ]
}

full_text_policy = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [{"path": "/text", "language": "en-US"}],
}

# Indexing policy (as provided by the user)
indexing_policy = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}, {"path": "/embedding/*"}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    "fullTextIndexes": [{"path": "/text"}],
}

# Example Usage (Optional - for testing)
if __name__ == "__main__":
    # Ensure COSMOS_CONNECTION_STRING is set in your environment variables
    if not COSMOS_CONNECTION_STRING:
        print("Error: COSMOS_CONNECTION_STRING environment variable not set.")
    else:
        client = SimpleCosmosClient(
            connection_string=COSMOS_CONNECTION_STRING,
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
            partition_key_path=PARTITION_KEY_PATH,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
        )

        client.connect()

        if client.database_client:
            # Example: Create the container

            # print("\nAttempting to create container...")

            # Example: Delete the container (use with caution!)
            # print("\nAttempting to delete container...")
            # client.delete_container(CONTAINER_NAME)

            # client.create_knowledge_chunks_container()

            query_term = "gemini"

            query = (
                f"SELECT TOP 5 * FROM c where FullTextContains(c.text, '{query_term}')"
            )
            print(f"Query: {query}")
            results = []
            try:
                # Assuming query_items returns an iterable; adjust if using async client
                for item in client.container_client.query_items(
                    query=query, enable_cross_partition_query=True
                ):
                    results.append(item)
                print(f"Found {len(results)} results for vector search.")
                # return results[:top_k]  # Return top_k results

            except exceptions.CosmosHttpResponseError as e:
                print(f"Error during vector search: {e}")
                results = []

            for r in results:

                def extract_windows(context, query, N):
                    context_lower = context.lower()
                    query_lower = query.lower()
                    positions = [
                        i
                        for i in range(len(context))
                        if context_lower.startswith(query_lower, i)
                    ]
                    return [
                        context[max(0, i - N) : i + len(query) + N] for i in positions
                    ]

                windows = extract_windows(r["text"], query_term, 50)
                print(windows)
