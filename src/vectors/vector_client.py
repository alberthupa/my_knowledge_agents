from dotenv import load_dotenv
import os
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


load_dotenv()




class DefaultEmbeddings:
    def set_embeddings(self):
        embeddings = AzureOpenAIEmbeddings(
            deployment="ada-002",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )
        return embeddings


class VectorStore:
    def __init__(self, name):
        self.name = name

    def load_vector_store(self, embeddings):
        if embeddings is None:
            print("No embeddings provided")
        else:
            if os.path.isdir(self.name):
                return FAISS.load_local(
                    self.name, embeddings, allow_dangerous_deserialization=True
                )
            else:
                return None
                
    def save_or_update_vector_store(self, documents, embeddings):
        """Create a new vector store or update existing one with documents"""
        import shutil
        
        existing_db = self.load_vector_store(embeddings)
        
        if existing_db is None:
            # Create new vector store
            new_db = FAISS.from_documents(documents, embeddings)
            new_db.save_local(self.name)
            return new_db
        else:
            # Update existing vector store
            new_db = FAISS.from_documents(documents, embeddings)
            existing_db.merge_from(new_db)
            shutil.rmtree(self.name)
            existing_db.save_local(self.name)
            return existing_db

if __name__ == "__main__":
    db_path = "vector_database"
    
    if not os.path.exists(db_path):
        print(f"No vector database found at {db_path}")
    else:
        try:
            embeddings = DefaultEmbeddings().set_embeddings()
            vector_store = VectorStore(db_path)
            loaded_vector_store = vector_store.load_vector_store(embeddings)
            
            if loaded_vector_store:
                print("Vector store loaded successfully.")
            else:
                print("Failed to load vector store - database may be corrupted or incompatible.")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")