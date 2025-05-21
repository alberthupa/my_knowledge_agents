import os
import argparse
import re
from datetime import datetime
from azure.cosmos import exceptions
from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import Dict, List
from src.vectors.cosmos_client import SimpleCosmosClient
from src.llms.basic_agent import BasicAgent
import json
import hashlib


from tqdm import tqdm
import time


"""
https://chatgpt.com/c/6824f32a-2c70-800a-8080-e55fd2007674
https://gemini.google.com/app/362b4f461787e386
"""


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
PARTITION_KEY_PATH = "/id"


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()
pieces = cosmos_client.database_client.get_container_client("knowledge-pieces")
chunks = cosmos_client.database_client.get_container_client("knowledge-chunks")


last_date = list(pieces.query_items(
    query="SELECT VALUE max(c.chunk_date) FROM c WHERE c.source = 'gmail_newsletter'",
    enable_cross_partition_query=True,
))[0]

'''
last_notes = list(chunks.query_items(
    query=f"SELECT c.id, c.chunk_date, c.subject, c.text FROM c WHERE c.chunk_date >= '{last_date}'",
    enable_cross_partition_query=True,
))

for i in last_notes:
    print(i)
'''

last_notes = list(pieces.query_items(
    query=f"SELECT c.id, c.headline FROM c WHERE c.chunk_date >= '{last_date}'",
    enable_cross_partition_query=True,
))

for i in last_notes:
    print(i)