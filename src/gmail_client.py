import os.path
import pickle
import base64
from datetime import datetime, timedelta
from email import message_from_bytes

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

"""
However, even if Zapier's dedicated "Azure Functions" action has limitations or complexities, there's a very common and flexible way to make Zapier talk to an Azure Function: __using Zapier's generic "Webhooks" action.__

Here's how that would work:

1. __Azure Function Setup:__

   - Your Azure Function would need to be configured with an __HTTP Trigger__. This gives your function a unique URL that can be called over the internet.

   - __Authentication for the HTTP Trigger:__ You can secure this HTTP endpoint in a few ways:

     - __Function Key (Recommended for simplicity with Zapier):__ Azure Functions can generate API keys (called "function keys"). You would include this key in the request from Zapier to authenticate. This is the most straightforward method for this use case.
     - __Azure AD Authentication:__ This is more advanced and involves setting up app registrations in Azure AD. While more secure for enterprise scenarios, it can be more complex to configure with external services like Zapier if they don't have a tailored integration for it. Given Zapier's message, starting with a Function Key is likely easier.

2. __Zapier Workflow Setup:__

   - __Trigger:__ "New email in Gmail" (with your label and any other criteria). This part remains the same.

   - __Action:__ Instead of looking for a specific "Azure Functions" action, you would use Zapier's __"Webhooks"__ action (often labeled as "POST", "GET", "PUT", or "Custom Request" within the Webhooks app by Zapier).

     - __URL:__ You'd paste the HTTP trigger URL of your Azure Function here.
     - __Payload Type:__ Usually JSON.
     - __Data:__ You would map the data from the Gmail trigger (e.g., email subject, body, sender, date, attachment info) into the body of the HTTP request that Zapier will send to your Azure Function.
     - __Headers:__ If you're using a Function Key for authentication, you would add a custom header here. For example, Azure Functions often look for a header like `x-functions-key` with the value of your function key.

__In summary, the plan would be:__

- __For Historical Emails (older than a certain date):__
  - Run the `gmail_client.py` script (that we've already worked on) __locally one time__. This allows you to handle the interactive OAuth consent and process the existing batch of emails.

- __For New Emails (moving forward):__

  1. Set up your Azure Function with an HTTP trigger, secured with a Function Key.

  2. In Zapier:

     - Trigger: New Gmail email matching your label.
     - Action: Use the "Webhooks by Zapier" app to send a POST request to your Azure Function's URL, including the email data in the request body and the Function Key in the headers.

  3. Your Azure Function receives this data and performs the necessary processing.

This approach bypasses any complexities with Zapier's direct Azure Function connector and uses a standard, widely supported method (HTTP requests/webhooks) for communication.

Does this revised plan using Zapier's Webhooks action to call your Azure Function sound like a good direction?

"""


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = (
    "gcp_credentials.json"  # Expects credentials.json in the same directory
)
TOKEN_FILE = "token.json"


def get_gmail_service():
    """Shows basic usage of the Gmail API.
    Handles OAuth 2.0 authentication and returns the Gmail API service object.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}")
                # Force re-authentication if refresh fails
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Error: Credentials file '{CREDENTIALS_FILE}' not found. "
                    "Please download it from Google Cloud Console and place it in the same directory."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # Use run_local_server for a better user experience if possible
            # Fallback to run_console if local server fails or not preferred
            try:
                creds = flow.run_local_server(port=0)
            except (
                OSError
            ):  # Handle cases where port might be in use or browser cannot be opened
                print(
                    "Could not start local server for OAuth, falling back to console."
                )
                creds = flow.run_console()

        # Save the credentials for the next run
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    try:
        service = build("gmail", "v1", credentials=creds)
        print("Gmail API service created successfully.")
        return service
    except HttpError as error:
        print(f"An error occurred building the service: {error}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


class GmailClient:
    """Client to interact with the Gmail API."""

    def __init__(self, service):
        """Initializes the GmailClient.

        Args:
            service: An authorized Gmail API service object.
        """
        if service is None:
            raise ValueError("Gmail API service object cannot be None.")
        self.service = service

    def list_messages(self, user_id="me", query=""):
        """Lists messages matching the query.

        Args:
            user_id: User's email address. The special value 'me' can be used
                     to indicate the authenticated user.
            query: String used to filter messages returned. Eg.- 'from:user@some_domain.com'
                   rfc822msgid: is required for message IDs.

        Returns:
            List of messages matching the criteria.
        """
        try:
            response = (
                self.service.users().messages().list(userId=user_id, q=query).execute()
            )
            messages = []
            if "messages" in response:
                messages.extend(response["messages"])

            # Handle pagination if there are more results
            while "nextPageToken" in response:
                page_token = response["nextPageToken"]
                response = (
                    self.service.users()
                    .messages()
                    .list(userId=user_id, q=query, pageToken=page_token)
                    .execute()
                )
                if "messages" in response:
                    messages.extend(response["messages"])
                else:
                    break  # Exit loop if no messages found in subsequent pages
            return messages
        except HttpError as error:
            print(f"An error occurred during message listing: {error}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during message listing: {e}")
            return None

    def get_message_details(self, user_id="me", msg_id=""):
        """Gets message details (metadata and snippet).

        Args:
            user_id: User's email address. The special value 'me' can be used
                     to indicate the authenticated user.
            msg_id: The ID of the message to retrieve.

        Returns:
            Message details including headers and snippet. Returns None if an error occurs.
        """
        try:
            # Request 'metadata' format for headers and snippet, 'full' for more details,
            # or 'raw' for the raw email content.
            message = (
                self.service.users()
                .messages()
                .get(userId=user_id, id=msg_id, format="metadata")
                .execute()
            )
            return message
        except HttpError as error:
            print(f"An error occurred getting message details: {error}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred getting message details: {e}")
            return None

    def download_emails(self, label_name, min_date_str):
        """Downloads emails with a specific label newer than a given date.

        Args:
            label_name (str): The name of the label to filter by (e.g., 'INBOX', 'MyLabel').
                              Can also be a label ID.
            min_date_str (str): The minimum date in 'YYYY-MM-DD' format. Emails newer
                                than this date (exclusive) will be fetched.
        """
        try:
            # Validate date format
            datetime.strptime(min_date_str, "%Y-%m-%d")
            # Format date for Gmail query (YYYY/MM/DD)
            query_date = min_date_str.replace("-", "/")
            query = f"label:{label_name} after:{query_date}"
            print(f"Searching for emails with query: '{query}'")

            messages = self.list_messages(query=query)

            if not messages:
                print("No messages found matching the criteria.")
                return

            print(f"Found {len(messages)} messages. Fetching details...")

            for i, msg_summary in enumerate(messages):
                msg_id = msg_summary["id"]
                print(f"\n--- Email {i+1}/{len(messages)} (ID: {msg_id}) ---")
                details = self.get_message_details(msg_id=msg_id)

                if details:
                    payload = details.get("payload", {})
                    headers = payload.get("headers", [])
                    subject = next(
                        (h["value"] for h in headers if h["name"] == "Subject"), "N/A"
                    )
                    from_addr = next(
                        (h["value"] for h in headers if h["name"] == "From"), "N/A"
                    )
                    date_str = next(
                        (h["value"] for h in headers if h["name"] == "Date"), "N/A"
                    )
                    snippet = details.get("snippet", "N/A")

                    print(f"  Subject: {subject}")
                    print(f"  From: {from_addr}")
                    print(f"  Date: {date_str}")
                    print(f"  Snippet: {snippet}")

                    # --- Optional: Get Full Content or Attachments ---
                    # To get the full raw email content (including body and attachments):
                    # 1. Change format in get_message_details to 'raw'.
                    # 2. Decode the base64url encoded string:
                    #    if 'raw' in details:
                    #        raw_data = base64.urlsafe_b64decode(details['raw'].encode('ASCII'))
                    #        email_message = message_from_bytes(raw_data)
                    #        # Now you can parse email_message using the 'email' library
                    #        # e.g., walk through parts to find body and attachments
                    #        print("Raw content fetched (length):", len(raw_data))

                    # To specifically download attachments:
                    # 1. Change format in get_message_details to 'full'.
                    # 2. Iterate through message parts in details['payload']['parts'].
                    # 3. If a part has a 'filename' and 'body.attachmentId', use:
                    #    attachment = service.users().messages().attachments().get(
                    #        userId='me', messageId=msg_id, id=part['body']['attachmentId']
                    #    ).execute()
                    #    file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
                    #    # Save file_data to a file named part['filename']
                    #    print(f"  Attachment found: {part['filename']}")

                else:
                    print("  Could not retrieve details for this message.")

        except ValueError:
            print(
                f"Error: Invalid date format '{min_date_str}'. Please use YYYY-MM-DD."
            )
        except Exception as e:
            print(f"An unexpected error occurred during email download: {e}")


if __name__ == "__main__":
    print("Starting Gmail client...")
    try:
        gmail_service = get_gmail_service()
        if gmail_service:
            client = GmailClient(gmail_service)

            # --- Example Usage ---
            # Set the label and date range you want to query
            target_label = "INBOX"  # Replace with your desired label name or ID
            # Calculate date 7 days ago
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            target_min_date = (
                seven_days_ago  # Replace with your desired start date 'YYYY-MM-DD'
            )

            print(
                f"\nAttempting to download emails from label '{target_label}' after {target_min_date}..."
            )
            client.download_emails(target_label, target_min_date)
            print("\nEmail download process finished.")
        else:
            print("Failed to create Gmail service. Exiting.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)  # Print the specific error from get_gmail_service
    except Exception as main_error:
        print(f"An error occurred in the main execution block: {main_error}")
