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
"""
how to get refresh token
Get the credentials.json file from https://console.cloud.google.com/apis/credentials
redirect_uri → should be http://localhost:8080/


from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=8080)

print("ACCESS TOKEN:", creds.token)
print("REFRESH TOKEN:", creds.refresh_token)

"""


creds = Credentials(
    token=None,
    refresh_token=os.environ.get("GMAIL_REFRESH_TOKEN"),
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.environ.get("GMAIL_CLIENT_ID"),
    client_secret=os.environ.get("GMAIL_CLIENT_SECRET"),
    scopes=["https://www.googleapis.com/auth/gmail.readonly"],
)
creds.refresh(Request())

service = build("gmail", "v1", credentials=creds)
"""
how to get labels
labels = service.users().labels().list(userId="me").execute()
for label in labels["labels"]:
    print(label["name"], "→", label["id"])

"""


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
        "subject": subject,
        "from": sender,
        "date": date,
        "content": cleaned_body[:10],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
