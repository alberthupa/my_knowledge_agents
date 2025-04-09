from dotenv import load_dotenv
import os
#import re
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
import json
import uuid
#import networkx as nx
#import matplotlib.pyplot as plt

from src.vectors.vector_client import VectorStore


load_dotenv()
db_path = "vector_database"

class Neo4jDriver:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI")
        self.user = "neo4j"
        self.password = os.environ.get("NEO4J_PASSWORD")
        self.driver = None

    def __enter__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.close()
            self.driver = None

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = (
                session.run(query, parameters) if parameters else session.run(query)
            )
            return list(result)
        
    def get_graph_schema(self):
        try:
            graph = Neo4jGraph(
                url=os.environ.get("NEO4J_URI"),
                username="neo4j",
                password=os.environ.get("NEO4J_PASSWORD"),
            )
            return graph.schema
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def is_empty(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count").single()
            return result["count"] == 0

    def describe(self):
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            nodes = session.run("MATCH (n) RETURN n LIMIT 3").values()
            rels = session.run("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 3").values()
            
            # Get counts by source_uri for nodes
            nodes_by_source = session.run(
                "MATCH (n) WHERE n.source_uri IS NOT NULL "
                "RETURN n.source_uri AS source, count(n) AS count "
                "ORDER BY count DESC"
            ).data()
            
            # Get counts by source_uri for relationships
            rels_by_source = session.run(
                "MATCH ()-[r]->() WHERE r.source_uri IS NOT NULL "
                "RETURN r.source_uri AS source, count(r) AS count "
                "ORDER BY count DESC"
            ).data()

        print(f"Nodes: {node_count}, Relationships: {rel_count}")
        print("First 3 nodes:")
        for node in nodes:
            node_str = str(node[0])
            print(f"{node_str[:50]}..." if len(node_str) > 50 else node_str)
        print("First 3 relationships:")
        for rel in rels:
            rel_str = str(rel)
            print(f"{rel_str[:50]}..." if len(rel_str) > 50 else rel_str)
            
        print("\nNodes by source_uri:")
        for item in nodes_by_source:
            print(f"  {item['source']}: {item['count']}")
        
        print("Relationships by source_uri:")
        for item in rels_by_source:
            print(f"  {item['source']}: {item['count']}")

    def import_from_json(self, json_file_path):
        """
        Import nodes and edges from a JSON file into Neo4j.
        All nodes will be created with the label 'Term'.
        Includes source_uri and uuid as properties for both nodes and relationships.
        
        :param json_file_path: Path to the JSON file containing nodes and edges data
        :return: Summary of the import operation
        """
        try:
            # Load JSON data
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            # Create unique constraint if it doesn't exist
            self.run_query("CREATE CONSTRAINT unique_term_name IF NOT EXISTS FOR (n:Term) REQUIRE n.name IS UNIQUE")
            
            # Process nodes
            nodes_count = 0
            if 'nodes' in data:
                for node_name, node_data in data.get('nodes', {}).items():
                    # Include source_uri and uuid as properties of the node
                    properties = {
                        'name': node_name,
                        'source_uri': node_data.get('source_uri', ''),
                        'uuid': str(uuid.uuid4())  # Add UUID as a unique identifier
                    }
                    
                    # Add all attributes to properties
                    for attr_key, attr_value in node_data.get('attributes', {}).items():
                        properties[attr_key] = attr_value
                    
                    # Create node with all properties including source_uri and uuid
                    query = (
                        "MERGE (n:Term {name: $name}) "
                        "SET n += $properties"
                    )
                    self.run_query(query, parameters={'name': node_name, 'properties': properties})
                    nodes_count += 1
            
            # Process edges
            edges_count = 0
            if 'edges' in data:
                for edge in data.get('edges', []):
                    source = edge.get('source')
                    target = edge.get('target')
                    relation = edge.get('relation')
                    source_uri = edge.get('source_uri', '')
                    
                    if source and target and relation:
                        # Create relationship with source_uri and uuid as properties
                        query = (
                            f"MATCH (source:Term {{name: $source}}), (target:Term {{name: $target}}) "
                            f"MERGE (source)-[r:`{relation}` {{source_uri: $source_uri, uuid: $uuid}}]->(target)"
                        )
                        self.run_query(query, parameters={
                            'source': source, 
                            'target': target, 
                            'source_uri': source_uri,
                            'uuid': str(uuid.uuid4())  # Add UUID for the relationship
                        })
                        edges_count += 1
            
            return {
                'success': True,
                'nodes_imported': nodes_count,
                'edges_imported': edges_count
            }
            
        except Exception as e:
            print(f"Error importing data from JSON: {e}")
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    with Neo4jDriver() as driver:


        #driver.clear_database()
        vector_store = VectorStore(db_path)
        vector_store.drop_vector_store()
        schema = driver.get_graph_schema()
        if schema:
            print("Graph schema retrieved successfully.")
            print(schema)
        else:
            print("Failed to retrieve graph schema.")

        driver.describe()

        # driver.import_from_json("/home/albert/python_projects/my_knowledge_agents/tmp_knowledge_graph/2503.24364v1.pdf.json")
        #embedder = GraphEmbedder()
        #embedder.embed_graph()