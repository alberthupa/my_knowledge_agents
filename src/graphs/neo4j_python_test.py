from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()

URI = os.environ.get("NEO4J_URI")  # Use environment variable if available
AUTH = ("neo4j", os.environ.get("NEO4J_PASSWORD"))  # Replace 'your_password' with your actual password

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")