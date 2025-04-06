import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.graphs.graph_client import Neo4jDriver
from src.vectors.vector_client import DefaultEmbeddings, VectorStore

name_of_vector_store = "vector_database"

embeddings = DefaultEmbeddings().set_embeddings()
vector_store = VectorStore(name_of_vector_store)
loaded_vector_store = vector_store.load_vector_store(embeddings)

with Neo4jDriver() as neo4jdriver:
    results = neo4jdriver.run_query("match (n) return n, labels(n) as labels")

# Collect all documents first
all_documents = []
for node in results:
    node_id = node["n"].get("uuid")
    node_name = node["n"].get("name")
    node_label = node["labels"][0]
    
    # Create node_content by concatenating all properties except uuid
    properties = []
    for key, value in node["n"].items():
        if key not in ["uuid", "source_uri"]:  # Skip uuid as it's already extracted
            # properties.append(f"{key}: {value}")
            properties.append(f"{value} ")
    node_content = "; ".join(properties)
    
    # print(node_content)

    metadata = {}
    if node_id:
        metadata["my_id"] = node_id
    if node_name:
        metadata["name"] = node_name
    if node_label:
        metadata["label"] = node_label

    list_of_documents = [
        Document(page_content=f"{node_name}: node_description", metadata=metadata)
    ]

    # Now save the documents to FAISS
    if loaded_vector_store is None:
        # Create a new vector store
        faiss_db = FAISS.from_documents(list_of_documents, embeddings)
        faiss_db.save_local(name_of_vector_store)
        print(f"Created new vector store with {len(list_of_documents)} documents")
    else:
        # Update existing vector store
        new_vector_db = FAISS.from_documents(list_of_documents, embeddings)
        loaded_vector_store.merge_from(new_vector_db)
        shutil.rmtree(name_of_vector_store)
        loaded_vector_store.save_local(name_of_vector_store)
        print(f"Updated vector store with {len(list_of_documents)} new documents")
