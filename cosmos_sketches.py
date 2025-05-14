import os
import argparse
import re
from datetime import datetime
from azure.cosmos import exceptions
from src.vectors.cosmos_client import CosmosClient

# Configuration variables
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"  # Replace with your database name if different
CONTAINER_NAME = "knowledge-chunks"  # Replace with your container name if different
PARTITION_KEY_PATH = "/id"  # Replace with your partition key path if different


def simplify_filename(subject, date_str):
    """Simplifies subject and date for use in a filename."""
    # Format date as YYYY-MM-DD
    try:
        # Assuming date_str is in a format that can be parsed by datetime
        date_obj = datetime.fromisoformat(
            date_str.replace("Z", "+00:00")
        )  # Handle potential 'Z' suffix
        formatted_date = date_obj.strftime("%Y-%m-%d")
    except ValueError:
        formatted_date = "unknown_date"  # Fallback if date parsing fails

    # Simplify subject: replace spaces with underscores, remove non-alphanumeric (except underscores)
    simplified_subject = re.sub(r"\s+", "_", subject)
    simplified_subject = re.sub(
        r"[^\w-]", "", simplified_subject
    )  # Allow hyphens for potential future use
    simplified_subject = simplified_subject.strip(
        "_"
    )  # Remove leading/trailing underscores

    return f"{formatted_date}_{simplified_subject}"


def retrieve_last_n_notes(client: CosmosClient, n: int):
    """Retrieves the last N notes from Cosmos DB."""
    if not client.container_client:
        print("Cosmos DB container client not available.")
        return []

    # Query to get the last N items ordered by chunk_date descending
    query = f"SELECT TOP {n} c.chunk_date, c.subject, c.text FROM c ORDER BY c.chunk_date DESC"
    print(f"Executing query: {query}")

    results = []
    try:
        # Assuming query_items returns an iterable
        for item in client.container_client.query_items(
            query=query, enable_cross_partition_query=True
        ):
            results.append(item)
        print(f"Found {len(results)} notes.")
        return results

    except exceptions.CosmosHttpResponseError as e:
        print(f"Error during query: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during query: {e}")
        return []


def save_notes_to_files(notes, output_dir="notes"):
    """Saves retrieved notes to separate text files."""
    if not notes:
        print("No notes to save.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving notes to directory: {output_dir}")

    for note in notes:
        subject = note.get("subject", "no_subject")
        chunk_date = note.get("chunk_date", "unknown_date")
        text = note.get("text", "")

        filename_base = simplify_filename(subject, chunk_date)
        filename = os.path.join(output_dir, f"{filename_base}.txt")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Subject: {subject}\n")
                f.write(f"Date: {chunk_date}\n\n")
                f.write(text)
            print(f"Saved note to {filename}")
        except IOError as e:
            print(f"Error saving note to {filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve and save last N notes from Cosmos DB."
    )
    parser.add_argument("n", type=int, help="Number of last notes to retrieve.")

    args = parser.parse_args()
    num_notes_to_retrieve = args.n

    if not COSMOS_CONNECTION_STRING:
        print("Error: COSMOS_CONNECTION_STRING environment variable not set.")
    else:
        # Note: vector_embedding_policy and indexing_policy are not needed for this simple retrieval
        # but the SimpleCosmosClient constructor requires them. Using dummy values or None if allowed.
        # Based on the provided SimpleCosmosClient, they are required.
        # I will use the values provided in the cosmos_client.py file.
        from src.vectors.cosmos_client import (
            vector_embedding_policy,
            indexing_policy,
            full_text_policy,
        )

        client = CosmosClient(
            connection_string=COSMOS_CONNECTION_STRING,
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
            partition_key_path=PARTITION_KEY_PATH,
            vector_embedding_policy=vector_embedding_policy,  # Using policy from cosmos_client.py
            indexing_policy=indexing_policy,  # Using policy from cosmos_client.py
        )

        client.connect()

        if client.container_client:
            notes = retrieve_last_n_notes(client, num_notes_to_retrieve)
            save_notes_to_files(notes)
        else:
            print("Failed to connect to Cosmos DB or get container client.")
