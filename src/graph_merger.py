import json
import os
import uuid
from langchain_core.documents import Document
from src.graphs.graph_client import Neo4jDriver
from src.vectors.vector_client import DefaultEmbeddings, VectorStore
from tqdm import tqdm  # Import tqdm for progress bars

class GraphMerger:
    def __init__(self, vector_store_name="vector_database", similarity_threshold=0.85):  # Increased threshold for better matching
        self.vector_store_name = vector_store_name
        self.similarity_threshold = similarity_threshold
        self.embeddings = DefaultEmbeddings().set_embeddings()
        self.vector_store = VectorStore(self.vector_store_name)
        self.loaded_vector_store = self.vector_store.load_vector_store(self.embeddings)
        # Track similarity scores for debugging
        self.similarity_scores = []
        
    def merge_from_json(self, json_file_path):
        """
        Merge nodes and relationships from a JSON file into the existing graph.
        For each node, check if a similar node already exists in the vector store.
        If it does, use that node's ID for creating relationships.
        If not, create a new node and embed it.
        
        :param json_file_path: Path to the JSON file containing nodes and edges data
        :return: Summary of the merge operation
        """
        try:
            # Reset similarity scores
            self.similarity_scores = []
            
            # Load JSON data
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            # Extract source_uri from JSON file
            json_source_uri = os.path.basename(json_file_path)
            print(f"Processing JSON from source: {json_source_uri}")
            
            # Stats to track the merge operation
            stats = {
                'nodes_added': 0,
                'nodes_matched': 0,
                'edges_added': 0,
                'errors': []
            }
            
            # Map to store node name -> node_id mappings
            node_id_map = {}
            
            # Debug: Check if vector store is loaded
            if not self.loaded_vector_store:
                print("WARNING: Vector store is not loaded! All nodes will be added as new.")
            else:
                print(f"Vector store loaded with similarity threshold: {self.similarity_threshold}")
            
            # Process nodes
            if 'nodes' in data:
                total_nodes = len(data['nodes'])
                print(f"Processing {total_nodes} nodes from JSON...")
                
                # Use tqdm for progress bar
                for node_name, node_data in tqdm(data.get('nodes', {}).items(), total=total_nodes, desc="Processing nodes"):
                    # Set source_uri if not present
                    if 'source_uri' not in node_data:
                        node_data['source_uri'] = json_source_uri
                    
                    # Create node content for similarity comparison
                    node_content = self._create_node_content(node_name, node_data)
                    
                    # Check if a similar node exists
                    similar_node, similarity = self._find_similar_node(node_content)
                    
                    if similar_node:
                        # Use the existing node's ID
                        node_id_map[node_name] = similar_node.metadata['my_id']
                        stats['nodes_matched'] += 1
                    else:
                        # Create a new node and embed it
                        node_id = self._add_new_node(node_name, node_data)
                        node_id_map[node_name] = node_id
                        stats['nodes_added'] += 1
                
                # Print interim stats after node processing
                print(f"\nNode processing complete:")
                print(f"- Nodes added: {stats['nodes_added']}")
                print(f"- Nodes matched: {stats['nodes_matched']}")
            
            # Process edges
            if 'edges' in data:
                total_edges = len(data['edges'])
                print(f"\nProcessing {total_edges} relationships...")
                
                for edge in tqdm(data.get('edges', []), total=total_edges, desc="Processing relationships"):
                    source = edge.get('source')
                    target = edge.get('target')
                    relation = edge.get('relation')
                    # Use JSON filename as source_uri if not specified
                    source_uri = edge.get('source_uri', json_source_uri)
                    
                    if source in node_id_map and target in node_id_map:
                        # Add relationship
                        success = self._add_relationship(
                            source_id=node_id_map[source],
                            target_id=node_id_map[target],
                            relation=relation,
                            source_uri=source_uri
                        )
                        if success:
                            stats['edges_added'] += 1
                    else:
                        error_msg = f"Could not create relationship: {source}-[{relation}]->{target}"
                        stats['errors'].append(error_msg)
            
            # Debug: Analyze similarity scores
            if self.similarity_scores:
                avg_similarity = sum(self.similarity_scores) / len(self.similarity_scores)
                max_similarity = max(self.similarity_scores)
                min_similarity = min(self.similarity_scores)
                print(f"\nSimilarity score statistics:")
                print(f"- Average: {avg_similarity:.4f}")
                print(f"- Maximum: {max_similarity:.4f}")
                print(f"- Minimum: {min_similarity:.4f}")
                print(f"- Threshold: {self.similarity_threshold}")
                below_threshold = sum(1 for s in self.similarity_scores if s < self.similarity_threshold)
                print(f"- Scores below threshold: {below_threshold} out of {len(self.similarity_scores)}")
            
            return stats
            
        except Exception as e:
            import traceback
            print(f"Error in merge_from_json: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_node_content(self, node_name, node_data):
        """Create content string for a node to be used for similarity comparison"""
        properties = []
        for key, value in node_data.get('attributes', {}).items():
            properties.append(f"{key}: {value}")
        
        return f"{node_name}: {'; '.join(properties)}"
    
    def _find_similar_node(self, node_content):
        """Find a similar node in the vector store using similarity search"""
        if not self.loaded_vector_store:
            return None, 0.0
        
        # Perform similarity search
        try:
            similar_docs = self.loaded_vector_store.similarity_search_with_score(
                node_content, k=1
            )
            
            # Check if any results and if similarity is above threshold
            if similar_docs and len(similar_docs) > 0:
                doc, score = similar_docs[0]
                similarity = 1.0 - score  # Convert distance to similarity
                
                # Store similarity score for debugging
                self.similarity_scores.append(similarity)
                
                if similarity >= self.similarity_threshold:
                    return doc, similarity
                
                # Remove detailed debug printout for rejected matches
                
            return None, 0.0
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return None, 0.0
    
    def _add_new_node(self, node_name, node_data):
        """Add a new node to the graph database and embed it in the vector store"""
        # Create a UUID for the new node
        node_id = str(uuid.uuid4())
        
        # Prepare properties for Neo4j
        base_properties = {
            'name': node_name,
            'source_uri': node_data.get('source_uri', ''),
            'uuid': node_id
        }
        
        # Add all attributes to properties with type checking and name sanitization
        attribute_properties = {}
        for attr_key, attr_value in node_data.get('attributes', {}).items():
            # Sanitize property key: replace spaces and special chars with underscores
            sanitized_key = attr_key.replace(' ', '_').replace('-', '_')
            
            # Convert complex types to JSON strings to avoid Neo4j type errors
            if isinstance(attr_value, (dict, list)):
                attribute_properties[sanitized_key] = json.dumps(attr_value)
            else:
                attribute_properties[sanitized_key] = attr_value
        
        # Merge all properties
        all_properties = {**base_properties, **attribute_properties}
        
        # Create node in Neo4j using a simpler approach
        with Neo4jDriver() as driver:
            # Use Neo4j's UNWIND to set properties safely
            query = (
                "MERGE (n:Term {name: $name}) "
                "SET n.uuid = $uuid, n.source_uri = $source_uri "
            )
            
            # Set attributes separately to avoid query building issues
            driver.run_query(query, parameters=base_properties)
            
            # Add each attribute property individually if there are any
            if attribute_properties:
                for key, value in attribute_properties.items():
                    property_query = f"MATCH (n:Term {{name: $name}}) SET n.{key} = $value"
                    driver.run_query(property_query, parameters={'name': node_name, 'value': value})
            
        # Embed the node in vector store with original property names
        node_content = self._create_node_content(node_name, node_data)
        self._embed_node(node_name, node_content, node_id)
        
        return node_id
    
    def _embed_node(self, node_name, node_content, node_id):
        """Embed a node in the vector store"""
        metadata = {
            'my_id': node_id,
            'name': node_name,
            'label': 'Term'  # Default label used in Neo4j
        }
        
        document = Document(page_content=node_content, metadata=metadata)
        
        if self.loaded_vector_store:
            # Update existing vector store
            self.vector_store.save_or_update_vector_store([document], self.embeddings)
        else:
            # Create a new vector store
            self.loaded_vector_store = self.vector_store.save_or_update_vector_store([document], self.embeddings)
    
    def _add_relationship(self, source_id, target_id, relation, source_uri):
        """Add a relationship between two nodes using their UUIDs"""
        try:
            with Neo4jDriver() as driver:
                # Create relationship with source_uri and uuid as properties
                query = (
                    f"MATCH (source:Term {{uuid: $source_id}}), (target:Term {{uuid: $target_id}}) "
                    f"MERGE (source)-[r:`{relation}` {{source_uri: $source_uri, uuid: $uuid}}]->(target)"
                )
                driver.run_query(query, parameters={
                    'source_id': source_id, 
                    'target_id': target_id, 
                    'source_uri': source_uri,
                    'uuid': str(uuid.uuid4())  # Add UUID for the relationship
                })
                return True
        except Exception as e:
            print(f"Error adding relationship: {e}")
            return False

    def analyze_json_file(self, json_file_path, test_only=True):
        """
        Analyze a JSON file without actually merging it.
        Shows how many nodes would be matched vs. added with current threshold.
        
        :param json_file_path: Path to the JSON file
        :param test_only: If True, only analyzes without making changes
        :return: Analysis results
        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            matched = 0
            would_add = 0
            similarity_data = []
            
            if 'nodes' in data:
                print(f"Analyzing {len(data['nodes'])} nodes with threshold {self.similarity_threshold}...")
                
                # Add progress bar for analysis
                for node_name, node_data in tqdm(data.get('nodes', {}).items(), desc="Analyzing nodes"):
                    node_content = self._create_node_content(node_name, node_data)
                    similar_node, similarity = self._find_similar_node(node_content)
                    
                    if similar_node:
                        matched += 1
                        similarity_data.append({
                            'new_node': node_name, 
                            'matched_node': similar_node.metadata.get('name'),
                            'similarity': similarity
                        })
                    else:
                        would_add += 1
            
            # Sort similarity data from highest to lowest
            similarity_data.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Print top 10 matches
            print("\nTop 10 highest similarity matches:")
            for i, data in enumerate(similarity_data[:10]):
                print(f"{i+1}. '{data['new_node']}' matched with '{data['matched_node']}' (similarity: {data['similarity']:.4f})")
            
            # Print bottom 10 matches if we have more than 10
            if len(similarity_data) > 10:
                print("\nBottom 10 similarity matches:")
                for i, data in enumerate(similarity_data[-10:]):
                    print(f"{i+1}. '{data['new_node']}' matched with '{data['matched_node']}' (similarity: {data['similarity']:.4f})")
            
            # Calculate distribution of similarity scores
            if similarity_data:
                print("\nSimilarity score distribution:")
                thresholds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
                for threshold in thresholds:
                    count = sum(1 for d in similarity_data if d['similarity'] >= threshold)
                    print(f"- Score >= {threshold:.2f}: {count} ({count/len(similarity_data)*100:.1f}%)")
            
            return {
                'would_match': matched,
                'would_add': would_add,
                'total': matched + would_add,
                'match_percentage': matched / (matched + would_add) * 100 if (matched + would_add) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error analyzing JSON: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    merger = GraphMerger(similarity_threshold=0.85)  # Increased threshold
    json_path = "/home/albert/python_projects/my_knowledge_agents/tmp_knowledge_graph/2503.24235v1.pdf.json"
    
    if os.path.exists(json_path):
        # First analyze without merging
        print("Analyzing JSON file without merging...")
        analysis = merger.analyze_json_file(json_path)
        print(f"\nAnalysis results:")
        print(f"- Would match: {analysis.get('would_match', 0)} nodes")
        print(f"- Would add as new: {analysis.get('would_add', 0)} nodes")
        print(f"- Match percentage: {analysis.get('match_percentage', 0):.1f}%")
        
        # Ask user if they want to proceed with actual merge
        proceed = input("\nDo you want to proceed with the merge? (y/n): ").lower() == 'y'
        
        if proceed:
            result = merger.merge_from_json(json_path)
            print("\nMerge results:")
            print(f"- Nodes added: {result.get('nodes_added', 0)}")
            print(f"- Nodes matched: {result.get('nodes_matched', 0)}")
            print(f"- Edges added: {result.get('edges_added', 0)}")
            
            if result.get('errors', []):
                print("Errors:")
                for error in result['errors']:
                    print(f"  - {error}")
        else:
            print("Merge operation cancelled.")
    else:
        print(f"JSON file not found: {json_path}")
