import os
import argparse
import re
from datetime import datetime
from azure.cosmos import exceptions
from src.vectors.cosmos_client import SimpleCosmosClient
from src.llms.basic_agent import BasicAgent

# Configuration variables
COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"  # Replace with your database name if different
CONTAINER_NAME = "knowledge-chunks"  # Replace with your container name if different
PARTITION_KEY_PATH = "/id"  # Replace with your partition key path if different


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()

if cosmos_client.container_client:
    last_date = cosmos_client.get_last_newsletter_date()
    print(last_date)
    last_notes = cosmos_client.get_notes_from_day(last_date)
    # print(last_notes)
    last_notes_str = str(last_notes)

    last_notes_str = str(last_notes)
message = f"This is a content of last newslettters about AI. Summarize it in bullers, telling what news is the most important: {last_notes_str}. Apply nice markdown formatting."


llm_agent = BasicAgent()
ai_response = llm_agent.get_text_response_from_llm(
    llm_model_input="gemini-2.0-flash-exp",  # Changed keyword argument
    messages=message,
    code_tag=None,
)
ai_response = ai_response["text_response"]
print(ai_response)
