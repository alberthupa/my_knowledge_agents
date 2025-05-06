from dotenv import load_dotenv

import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from src.vectors.embeddings_clients import OpenAIEmbeddings

import os, json, re
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import html2text
from email.utils import parsedate_to_datetime
import base64


load_dotenv(override=True)

COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
CONTAINER_NAME = "test_container"
PARTITION_KEY_PATH = "/id"


# calculate embeddings
embeddings_result = OpenAIEmbeddings().get_openai_embedding("test")
calculated_embeddings = embeddings_result.data[0].embedding
print(calculated_embeddings)


# --- Cosmos DB Client ---


def get_cosmos_client(connection_string):
    parts = connection_string.split(";")
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

        print(
            f"Successfully connected to database '{database_name}' and container '{container_name}'."
        )
        return container
    except exceptions.CosmosResourceNotFoundError as e:
        # Check if the error is specifically for the database or container
        if f"dbs/{database_name}" in e.message:
            print(
                f"Error: Database '{database_name}' not found. Please ensure the database name is correct and exists."
            )
        elif f"colls/{container_name}" in e.message:
            print(
                f"Error: Container '{container_name}' not found in database '{database_name}'. Please ensure the container name is correct and exists."
            )
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
def upload_text_chunk(
    container,
    item_id,
    text_content,
    tags,
    vector_embedding=None,
    custom_properties=None,
):
    """Uploads a text chunk with tags and custom properties to the container."""
    document = {
        "id": item_id,
        "text": text_content,
        "tags": tags,
        "vector": vector_embedding,  # Placeholder for vector embedding
    }
    if custom_properties:
        document.update(custom_properties)
    try:
        container.upsert_item(document)
        print(f"Uploaded item with id: {item_id}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Error uploading item {item_id}: {e}")


cosmos_client = get_cosmos_client(COSMOS_CONNECTION_STRING)
if cosmos_client:
    container_client = get_container_client(
        cosmos_client, DATABASE_NAME, CONTAINER_NAME
    )

    if container_client:
        # Example of uploading a text chunk with custom properties
        custom_props = {
            "source_file": "example_document.txt",
            "processing_date": "2025-05-04",
            "category": "documentation",
        }
        upload_text_chunk(
            container_client,
            item_id="doc2",
            text_content="This is the first document about Azure Cosmos DB blah blah",
            tags=["azure", "cosmosdb"],
            custom_properties=custom_props,
        )


# --- Google API Client ---
gmail_creds = Credentials(
    token=None,
    refresh_token=os.environ.get("GMAIL_REFRESH_TOKEN"),
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.environ.get("GMAIL_CLIENT_ID"),
    client_secret=os.environ.get("GMAIL_CLIENT_SECRET"),
    scopes=["https://www.googleapis.com/auth/gmail.readonly"],
)
gmail_creds.refresh(Request())

service = build("gmail", "v1", credentials=gmail_creds)


def clean_markdown(md):
    md = re.sub(r"!\[\]\(https://track\.[^\)]+\)", "", md)
    md = re.sub(r"\[View in browser\]\([^\)]+\)", "", md, flags=re.I)
    md = re.sub(
        r"\[.*?\]\((https?://(?:track\.aisecret\.us|click\.convertkit-mail2\.com)[^\)]+)\)",
        "",
        md,
    )
    md = re.sub(r"\[!\[\]\([^\)]+\)\]\([^\)]+\)", "", md)
    md = re.sub(r"^\s*> \[.*?SPONSORED.*?\]\(.*?\)\s*$", "", md, flags=re.M)
    md = re.sub(r"\*\s?\*\s?\*", "", md)
    md = re.sub(r"---+", "", md)
    md = re.sub(r"\|\s?.*?\s?\|", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r" +\n", "\n", md)
    md = re.sub(r"\n +", "\n", md)

    md = re.split(r"https://track\.aisecret\.us/track/unsubscribe\.do\?", md)[0].strip()
    # md = re.sub(r"\s+", " ", md).strip()  # single-line body
    return md


results = (
    service.users()
    .messages()
    .list(userId="me", labelIds=["Label_58"], maxResults=5)
    .execute()
)

for msg in results.get("messages", []):
    full_msg = (
        service.users()
        .messages()
        .get(userId="me", id=msg["id"], format="full")
        .execute()
    )

    headers = {h["name"]: h["value"] for h in full_msg["payload"]["headers"]}
    subject = headers.get("Subject", "")
    sender = headers.get("From", "")
    raw_date = headers.get("Date", "")
    try:
        date_obj = parsedate_to_datetime(raw_date)
        date = date_obj.date().isoformat()  # YYYY-MM-DD
    except Exception:
        date = raw_date  # fallback if parsing fails

    def find_html(part):
        if part.get("mimeType") == "text/html":
            return part.get("body", {}).get("data")
        for p in part.get("parts", []):
            found = find_html(p)
            if found:
                return found
        return None

    raw_html = find_html(full_msg["payload"])
    decoded_html = (
        base64.urlsafe_b64decode(raw_html.encode()).decode() if raw_html else ""
    )

    markdown = html2text.HTML2Text()
    markdown.ignore_links = False
    markdown_body = markdown.handle(decoded_html)
    cleaned_body = clean_markdown(markdown_body)

    output = {
        "source": "newsletters",
        "subject": subject,
        "from": sender,
        "date": date,
        "content": cleaned_body,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
