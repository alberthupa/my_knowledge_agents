import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

"""
# TODO: add
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode("your text here")
You can run it locally in the Azure Function using ONNX or PyTorch.

https://jina.ai/models/jina-embeddings-v3/

"""


class OpenAIEmbeddings:
    """
    sth = OpenAIEmbeddings().get_openai_embedding("test")
    print(sth.data[0].embedding)
    """

    def set_embeddings_client(self):
        return OpenAI()

    def get_openai_embedding(self, text):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client.embeddings.create(input=text, model="text-embedding-3-small")
