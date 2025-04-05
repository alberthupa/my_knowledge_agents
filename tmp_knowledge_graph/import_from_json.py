import json
from neo4j import GraphDatabase

def import_from_json(json_file_path, neo4j_driver):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create unique constraint if it doesn't exist
    with neo4j_driver.session() as session:
        session.run("CREATE CONSTRAINT unique_node_name IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE")
        
    # Process nodes
    with neo4j_driver.session() as session:
        for node_name, node_data in data.get('nodes', {}).items():
            # Include source_uri as a property of the node
            properties = {
                'name': node_name,
                'source_uri': node_data.get('source_uri', '')
            }
            
            # Add all attributes to properties
            for attr_key, attr_value in node_data.get('attributes', {}).items():
                properties[attr_key] = attr_value
                
            # Create node with all properties including source_uri
            query = (
                "MERGE (n:Node {name: $name}) "
                "SET n += $properties"
            )
            session.run(query, name=node_name, properties=properties)

    # Process edges
    with neo4j_driver.session() as session:
        for edge in data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            relation = edge.get('relation')
            source_uri = edge.get('source_uri', '')
            
            # Create relationship with source_uri as property
            query = (
                "MATCH (source:Node {name: $source}), (target:Node {name: $target}) "
                "MERGE (source)-[r:" + relation + " {source_uri: $source_uri}]->(target)"
            )
            session.run(query, source=source, target=target, source_uri=source_uri)

if __name__ == "__main__":
    # Example usage
    json_file_path = "path_to_json_file.json"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    import_from_json(json_file_path, driver)
    driver.close()