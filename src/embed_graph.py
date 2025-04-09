import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.graphs.graph_client import Neo4jDriver
from src.vectors.vector_client import DefaultEmbeddings, VectorStore


class GraphEmbedder:
    def __init__(self, vector_store_name="vector_database"):
        self.name_of_vector_store = vector_store_name
        self.embeddings = DefaultEmbeddings().set_embeddings()
        self.vector_store = VectorStore(self.name_of_vector_store)
        self.loaded_vector_store = self.vector_store.load_vector_store(self.embeddings)
    
    def embed_graph(self):
        """Embed all nodes from the graph into the vector store."""
        with Neo4jDriver() as neo4jdriver:
            results = neo4jdriver.run_query("match (n) return n, labels(n) as labels")
        
        # Track number of nodes processed
        nodes_processed = 0
        
        # Process each node
        for node in results:
            nodes_processed += 1
            self._process_and_store_node(node)
            
        return nodes_processed
    
    def _process_and_store_node(self, node):
        node_id = node["n"].get("uuid")
        node_name = node["n"].get("name")
        node_label = node["labels"][0]
        
        # Create node_content by concatenating all properties except uuid
        properties = []
        for key, value in node["n"].items():
            if key not in ["uuid", "source_uri"]:  # Skip uuid as it's already extracted
                properties.append(f"{value} ")
        node_content = "; ".join(properties)
        
        metadata = {}
        if node_id:
            metadata["my_id"] = node_id
        if node_name:
            metadata["name"] = node_name
        if node_label:
            metadata["label"] = node_label

        list_of_documents = [
            Document(page_content=f"{node_name}: {node_content}", metadata=metadata)
        ]

        # Now save the documents to FAISS
        if self.loaded_vector_store is None:
            # Create a new vector store
            faiss_db = FAISS.from_documents(list_of_documents, self.embeddings)
            faiss_db.save_local(self.name_of_vector_store)
            print(f"Created new vector store with {len(list_of_documents)} documents")
            self.loaded_vector_store = faiss_db
        else:
            # Update existing vector store
            new_vector_db = FAISS.from_documents(list_of_documents, self.embeddings)
            self.loaded_vector_store.merge_from(new_vector_db)
            shutil.rmtree(self.name_of_vector_store)
            self.loaded_vector_store.save_local(self.name_of_vector_store)
            print(f"Updated vector store with {len(list_of_documents)} new documents")
    
    def get_statistics(self):
        """Get basic statistics about the embedded graph."""
        if not os.path.exists(self.name_of_vector_store) or self.loaded_vector_store is None:
            return {"status": "empty", "count": 0}
        
        try:
            # Get document count
            count = len(self.loaded_vector_store.docstore._dict)
            
            # Get unique node labels
            labels = set()
            for doc_id, doc in self.loaded_vector_store.docstore._dict.items():
                if "label" in doc.metadata:
                    labels.add(doc.metadata["label"])
                    
            return {
                "status": "populated",
                "count": count,
                "labels": list(labels)
            }
        except:
            return {"status": "error", "count": 0}


if __name__ == "__main__":
    embedder = GraphEmbedder()
    embedder.embed_graph()
    stats = embedder.get_statistics()
    
    if stats["status"] == "empty" or stats["count"] == 0:
        print("There are no embeds in the vector store.")
    else:
        print(f"Graph Embedding Statistics:")
        print(f"- Total embedded nodes: {stats['count']}")
        if "labels" in stats and stats["labels"]:
            print(f"- Node types: {', '.join(stats['labels'])}")
        print(f"- Vector store path: {embedder.name_of_vector_store}")
