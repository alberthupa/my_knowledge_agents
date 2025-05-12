import os
import requests
# from bertopic import BERTopic
# from hdbscan.hdbscan_ import HDBSCAN
# from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP  # Import UMAP


# from bertopic import BERTopic
# from hdbscan import HDBSCAN
# from umap import UMAP
# from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer
# from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
import numpy as np
from top2vec import Top2Vec  # top2vec 1.0.36  pip install top2vec[sentence_encoders]

# --------- CONFIGURATION ---------
NOTES_DIR = "./notes/"


def get_txt_files(directory):
    """Return a list of .txt file paths in the given directory."""
    try:
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".txt")
        ]
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []


def read_documents(file_list):
    """Read the content of each file into a list of documents."""
    documents = []
    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents


def main():
    file_list = get_txt_files(NOTES_DIR)
    print(file_list)
    documents = read_documents(file_list)
    if not documents:
        print("No documents found. Exiting.")
        return

    # --- TOPIC DETECTION WITH TOP2VEC ---
    print("Fitting Top2Vec model to documents...")
    try:
        # You can adjust embedding_model and speed parameters as needed
        model = Top2Vec(
            documents,
            embedding_model="universal-sentence-encoder",  # or "distiluse-base-multilingual-cased"
            speed="learn",  # "deep-learn" for best quality, "learn" for speed
            workers=4,
        )
    except Exception as e:
        print(f"Error fitting Top2Vec: {e}")
        return

    # Print number of topics
    num_topics = model.get_num_topics()
    print(f"\nNumber of topics found: {num_topics}")

    # Print top words for each topic
    topic_words, word_scores, topic_nums = model.get_topics()
    for topic_idx, (words, scores, topic_num) in enumerate(
        zip(topic_words, word_scores, topic_nums)
    ):
        print(f"\nTopic {topic_num}:")
        for word, score in zip(words, scores):
            print(f"  {word} ({score:.3f})")

    # Show which documents belong to which topics
    print("\nDocument assignments:")
    doc_topics, doc_scores, doc_nums = model.get_documents_topics(
        list(range(len(documents)))
    )
    for idx, (topic, score) in enumerate(zip(doc_topics, doc_scores)):
        print(f"Document {idx}: Topic {topic} (Score: {score:.2f})")

    # Optional: visualize topics if matplotlib is installed
    try:
        import matplotlib.pyplot as plt

        model.generate_topic_wordcloud(topic_num=0)
        print("Wordcloud for Topic 0 displayed.")
    except Exception as e:
        print("Visualization requires matplotlib and a display environment.")


if __name__ == "__main__":
    main()
