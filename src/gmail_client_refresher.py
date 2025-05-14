import os, json, re
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import html2text
from email.utils import parsedate_to_datetime
import base64
from google_auth_oauthlib.flow import InstalledAppFlow


load_dotenv(override=True)

# how to get refresh token
# Get the credentials.json file from https://console.cloud.google.com/apis/credentials
# redirect_uri → should be http://localhost:8080/


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
creds = flow.run_local_server(port=8080)

print("ACCESS TOKEN:", creds.token)
print("REFRESH TOKEN:", creds.refresh_token)
