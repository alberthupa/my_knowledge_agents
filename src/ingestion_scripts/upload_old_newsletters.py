from dotenv import load_dotenv
import os
import json
import re
import base64
from datetime import datetime, timedelta, date as py_date  # Renamed to avoid conflict

from azure.cosmos import CosmosClient, exceptions
from src.vectors.embeddings_clients import OpenAIEmbeddings
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import html2text
from email.utils import parsedate_to_datetime

load_dotenv(override=True)

# --- Environment Variables & Constants ---
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
CONTAINER_NAME = "test_container"


GMAIL_REFRESH_TOKEN = os.environ.get("GMAIL_REFRESH_TOKEN")
GMAIL_CLIENT_ID = os.environ.get("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.environ.get("GMAIL_CLIENT_SECRET")

# --- OpenAI Embeddings Client ---
try:
    embeddings_client = OpenAIEmbeddings()
    print("OpenAIEmbeddings client initialized.")
except Exception as e:
    print(f"Error initializing OpenAIEmbeddings client: {e}")
    embeddings_client = None


# --- Cosmos DB Client ---
def get_cosmos_client(connection_string):
    """Initializes and returns a CosmosClient."""
    if not connection_string:
        print("Error: COSMOS_CONNECTION_STRING is not set.")
        return None
    try:
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
            raise ValueError(
                "Invalid connection string format. Ensure AccountEndpoint and AccountKey are present."
            )
        print(f"Attempting to connect to Cosmos DB at URI: {uri}")
        client = CosmosClient(uri, credential=key)
        # Test connection by listing databases (optional, can be slow)
        # list(client.list_databases())
        print("CosmosClient initialized successfully.")
        return client
    except ValueError as ve:
        print(f"Error initializing CosmosClient: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while initializing CosmosClient: {e}")
        return None


def get_container_client(client, database_name, container_name):
    """Gets a reference to the database and container."""
    if not client:
        print("Error: Cosmos client is not initialized.")
        return None
    try:
        database = client.get_database_client(database_name)
        print(f"Successfully got database client for '{database_name}'.")
        container = database.get_container_client(container_name)
        print(f"Successfully got container client for '{container_name}'.")
        # Test container read (optional)
        # container.read()
        print(
            f"Successfully connected to database '{database_name}' and container '{container_name}'."
        )
        return container
    except exceptions.CosmosResourceNotFoundError as e:
        if f"dbs/{database_name}" in str(e):
            print(f"Error: Database '{database_name}' not found.")
        elif f"colls/{container_name}" in str(e):
            print(
                f"Error: Container '{container_name}' not found in database '{database_name}'."
            )
        else:
            print(f"Error: A Cosmos DB resource was not found: {e}")
        return None
    except exceptions.CosmosHttpResponseError as e:
        print(f"An unexpected Cosmos DB HTTP error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting container client: {e}")
        return None


def check_item_exists(container, item_id):
    """Checks if an item with the given ID exists in the container."""
    if not container:
        print(f"Debug: Container not available for checking item {item_id}.")
        return False  # Cannot confirm, assume not exists or handle error
    try:
        print(f"Debug: Checking if item with id '{item_id}' exists in Cosmos DB...")
        container.read_item(
            item=item_id, partition_key=item_id
        )  # Assuming id is also the partition key
        print(f"Debug: Item with id '{item_id}' FOUND in Cosmos DB.")
        return True
    except exceptions.CosmosResourceNotFoundError:
        print(f"Debug: Item with id '{item_id}' NOT FOUND in Cosmos DB.")
        return False
    except exceptions.CosmosHttpResponseError as e:
        print(f"Debug: Error checking item '{item_id}': {e}. Assuming not found.")
        return False  # Or re-raise depending on desired error handling
    except Exception as e:
        print(
            f"Debug: Unexpected error checking item '{item_id}': {e}. Assuming not found."
        )
        return False


def upload_text_chunk(
    container,
    item_id,
    text_content,
    tags,
    vector_embedding=None,
    custom_properties=None,
):
    """Uploads a text chunk with tags and custom properties to the container."""
    if not container:
        print(f"Debug: Container not available for uploading item {item_id}.")
        return
    document = {
        "id": item_id,  # This will also be used as the partition key if PK is /id
        "text": text_content,
        "tags": tags,
        "embedding": vector_embedding,
    }
    if custom_properties:
        document.update(custom_properties)

    print(f"Debug: Preparing to upload document with id: {item_id}")
    # print(f"Debug: Document content (first 100 chars of text): {json.dumps(document, default=str)[:200]}...")

    try:
        container.upsert_item(document)
        print(f"Debug: Successfully uploaded/updated item with id: {item_id}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Debug: Error uploading item {item_id}: {e}")
        print(f"Debug: Failed document details: {json.dumps(document, default=str)}")
    except Exception as e:
        print(f"Debug: Unexpected error uploading item {item_id}: {e}")


# --- Google API Client ---
def get_gmail_service():
    """Initializes and returns the Gmail API service."""
    print("Debug: Initializing Gmail service...")
    if not all([GMAIL_REFRESH_TOKEN, GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET]):
        print(
            "Error: Gmail API credentials (refresh token, client ID, client secret) not fully set in .env."
        )
        return None
    try:
        gmail_creds = Credentials(
            token=None,
            refresh_token=GMAIL_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GMAIL_CLIENT_ID,
            client_secret=GMAIL_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        gmail_creds.refresh(Request())
        service = build("gmail", "v1", credentials=gmail_creds)
        print("Debug: Gmail service initialized successfully.")
        return service
    except Exception as e:
        print(f"Error initializing Gmail service: {e}")
        return None


def clean_markdown(md):
    """Cleans markdown text extracted from HTML emails."""
    # Remove tracking pixels and common unwanted elements
    md = re.sub(r"!\[\]\(https://track\.[^\)]+\)", "", md)
    md = re.sub(r"\[View in browser\]\([^\)]+\)", "", md, flags=re.I)
    md = re.sub(
        r"\[.*?\]\((https?://(?:track\.aisecret\.us|click\.convertkit-mail2\.com|mandrillapp\.com|clicks\.mlsend\.com|click\.sender\.net|t\.dripemail2\.com|click\.revue\.email|ct\.beehiiv\.com|clicks\.aweber\.com|hubspotlinks\.com|getresponse\.com|substack\.com|mailerlite\.com|sendgrid\.net|sparkpostmail\.com|amazonseS\.com)[^\)]+)\)",
        "",
        md,
    )  # Common tracking domains
    md = re.sub(
        r"\[!\[\]\([^\)]+\)\]\([^\)]+\)", "", md
    )  # Image links that are also links
    md = re.sub(
        r"^\s*> \[.*?SPONSORED.*?\]\(.*?\)\s*$", "", md, flags=re.M | re.I
    )  # Sponsored content blocks
    md = re.sub(r"\*\s?\*\s?\*", "", md)  # Horizontal rules like ***
    md = re.sub(r"---+", "", md)  # Other horizontal rules
    md = re.sub(r"\|\s?.*?\s?\|", "", md)  # Table-like structures (simple ones)

    # Remove unsubscribe links and everything after them (common pattern)
    unsubscribe_patterns = [
        r"https://track\.aisecret\.us/track/unsubscribe\.do\?",
        r"If you wish to stop receiving our emails.*?click here",
        r"To unsubscribe from this newsletter.*?click here",
        r"No longer want to receive these emails\?",
        r"Unsubscribe here",
        r"Manage your preferences",
        r"Update your profile",
    ]
    for pattern in unsubscribe_patterns:
        md = re.split(pattern, md, flags=re.I)[0].strip()

    md = re.sub(r"\n{3,}", "\n\n", md)  # Reduce multiple newlines to two
    md = re.sub(r" +\n", "\n", md)  # Trim trailing spaces from lines
    md = re.sub(r"\n +", "\n", md)  # Trim leading spaces from lines

    # Remove lines that are just image placeholders or decorative characters
    md = re.sub(r"^\s*\[image:.*?\]\s*$", "", md, flags=re.M | re.I)
    md = re.sub(
        r"^\s*!\[.*?\]\(.*?\)\s*$", "", md, flags=re.M | re.I
    )  # Standalone image markdown
    md = re.sub(
        r"^\s*[-=_\*#]{3,}\s*$", "", md, flags=re.M
    )  # Lines with only separators

    return md.strip()


def find_html_part(parts):
    """Recursively finds the HTML part in email message parts."""
    for part in parts:
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            return part["body"]["data"]
        if "parts" in part:
            html_data = find_html_part(part["parts"])
            if html_data:
                return html_data
    return None


# --- Main Processing Logic ---
def process_newsletters_for_date(
    target_date_str, gmail_service, container_client, embeddings_client_instance
):
    """Processes newsletters for a specific date."""
    print(f"Debug: Starting newsletter processing for date: {target_date_str}")

    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{target_date_str}'. Please use YYYY-MM-DD.")
        return

    # Gmail search query for the specific day
    # Gmail API uses timestamps, so we define a 24-hour window
    # For date YYYY/MM/DD, search from YYYY/MM/DD 00:00:00 to YYYY/MM/DD 23:59:59
    # Gmail search format is 'after:YYYY/MM/DD before:YYYY/MM/DD' (exclusive for before)
    # So, for a single day, use after:YYYY/MM/DD before:YYYY/MM/(DD+1)

    date_after = target_dt.strftime("%Y/%m/%d")
    date_before = (target_dt + timedelta(days=1)).strftime("%Y/%m/%d")
    query = f"after:{date_after} before:{date_before}"
    # Optional: Add label filter if newsletters are consistently labeled
    # query += " label:your-newsletter-label"
    print(f"Debug: Gmail search query: '{query}'")

    try:
        results = (
            gmail_service.users()
            .messages()
            .list(userId="me", q=query, labelIds=["Label_58"], maxResults=50)
            .execute()
        )  # Increased maxResults
    except Exception as e:
        print(f"Error querying Gmail: {e}")
        return

    messages = results.get("messages", [])
    print(f"Debug: Found {len(messages)} messages for {target_date_str}.")

    if not messages:
        print(f"Debug: No messages found for {target_date_str}. Exiting.")
        return

    for msg_summary in messages:
        msg_id = msg_summary["id"]
        print(f"\nDebug: Processing message ID: {msg_id}")

        if check_item_exists(container_client, msg_id):
            print(f"Debug: Message ID {msg_id} already exists in Cosmos DB. Skipping.")
            continue

        try:
            full_msg = (
                gmail_service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
        except Exception as e:
            print(f"Error fetching full message for ID {msg_id}: {e}")
            continue

        payload = full_msg.get("payload", {})
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

        subject = headers.get("Subject", "No Subject")
        sender = headers.get("From", "Unknown Sender")
        raw_date_header = headers.get("Date", "")

        try:
            parsed_date_obj = parsedate_to_datetime(raw_date_header)
            message_date_iso = parsed_date_obj.date().isoformat()  # YYYY-MM-DD
        except Exception:
            message_date_iso = (
                target_date_str  # Fallback to target date if parsing fails
            )

        print(f"Debug: Subject: {subject}")
        print(f"Debug: From: {sender}")
        print(f"Debug: Date: {message_date_iso}")

        html_data = None
        if payload.get("mimeType") == "text/html":
            html_data = payload.get("body", {}).get("data")
        elif "parts" in payload:
            html_data = find_html_part(payload["parts"])

        if not html_data:
            print(f"Debug: No HTML content found for message ID {msg_id}. Skipping.")
            continue

        try:
            decoded_html = base64.urlsafe_b64decode(html_data.encode("UTF-8")).decode(
                "UTF-8"
            )
        except Exception as e:
            print(f"Debug: Error decoding HTML for message ID {msg_id}: {e}. Skipping.")
            continue

        h = html2text.HTML2Text()
        h.ignore_links = False  # Keep links for context, cleaning will handle bad ones
        h.body_width = 0  # Don't wrap lines
        markdown_body = h.handle(decoded_html)
        cleaned_body = clean_markdown(markdown_body)

        if not cleaned_body.strip():
            print(f"Debug: Cleaned body is empty for message ID {msg_id}. Skipping.")
            continue

        print(
            f"Debug: Cleaned content (first 100 chars): {cleaned_body[:100].replace(chr(10), ' ')}..."
        )

        # Calculate embeddings
        if embeddings_client_instance:
            try:
                print(f"Debug: Calculating embeddings for message ID {msg_id}...")
                # Combine subject and body for embedding for better context
                text_to_embed = f"Subject: {subject}\n\n{cleaned_body}"
                embeddings_result = embeddings_client_instance.get_openai_embedding(
                    text_to_embed
                )
                calculated_embeddings = embeddings_result.data[0].embedding
                print(
                    f"Debug: Embeddings calculated successfully for message ID {msg_id}."
                )
            except Exception as e:
                print(f"Error calculating embeddings for message ID {msg_id}: {e}")
                calculated_embeddings = None  # Proceed without embeddings or skip
        else:
            print(
                "Debug: Embeddings client not available. Skipping embedding calculation."
            )
            calculated_embeddings = None

        custom_props = {
            "source": "gmail_newsletter",
            "subject": subject,
            "from_sender": sender,  # Renamed to avoid conflict if 'from' is a keyword
            "received_date": message_date_iso,  # Date from email header
            "processing_target_date": target_date_str,  # Date used for the query
            "gmail_message_id": msg_id,
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }

        upload_text_chunk(
            container_client,
            item_id=msg_id,  # Use Gmail message ID as the document ID
            text_content=cleaned_body,
            tags=["newsletter", sender, subject[:50]],  # Example tags
            vector_embedding=calculated_embeddings,
            custom_properties=custom_props,
        )
        print(f"Debug: Upload process initiated for message ID {msg_id}.")


if __name__ == "__main__":
    print("--- Starting Newsletter Ingestion Script (Debug Mode - Single Day) ---")

    # For debugging, process today's date.
    # You can change this to a specific date string like "2025-05-05"
    # TARGET_DATE = py_date.today().strftime("%Y-%m-%d")
    TARGET_DATE = "2025-05-05"  # Example: specific date

    print(f"Debug: Target processing date set to: {TARGET_DATE}")

    # Initialize clients
    gmail_service_instance = get_gmail_service()
    cosmos_client_instance = get_cosmos_client(COSMOS_CONNECTION_STRING)

    container_client_instance = None
    if cosmos_client_instance:
        container_client_instance = get_container_client(
            cosmos_client_instance, DATABASE_NAME, CONTAINER_NAME
        )

    if (
        gmail_service_instance and container_client_instance and embeddings_client
    ):  # Check embeddings_client too
        process_newsletters_for_date(
            TARGET_DATE,
            gmail_service_instance,
            container_client_instance,
            embeddings_client,
        )
    else:
        print(
            "Error: One or more services (Gmail, Cosmos DB, OpenAI Embeddings) could not be initialized. Exiting."
        )
        if not gmail_service_instance:
            print("Debug: Gmail service failed to initialize.")
        if not container_client_instance:
            print("Debug: Cosmos container client failed to initialize.")
        if not embeddings_client:
            print("Debug: OpenAI embeddings client failed to initialize.")

    print("--- Newsletter Ingestion Script Finished ---")
