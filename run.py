#!/usr/bin/env python3

import sys
import os
from src.ingestion_scripts.file_iterator import extract_kg_from_doc


# nth


def ingest_kg_from_script():
    """Main entry point for knowledge graph extraction"""
    # Hardcoded settings
    file_path = "sources/2503.24235v1.pdf"  # Path to your document
    # file_path = "sources/2503.24364v1.pdf"  # Path to your document
    llm_model = "gemini-2.0-flash"  # Primary LLM model
    secondary_llm_model = "gpt-4o"  # Secondary LLM model (can be None)
    max_tokens_per_window = 4000  # Maximum tokens per window (T)
    num_runs_per_function = 2  # Number of runs per function (N)
    overlap_tokens = 50  # Token overlap between windows (O)
    output_dir = "tmp_knowledge_graph"  # Directory to save the knowledge graph

    # Verify that the input file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return 1

    # Extract knowledge graph from the document
    knowledge_graph = extract_kg_from_doc(
        file_path=file_path,
        llm_model=llm_model,
        secondary_llm_model=secondary_llm_model,
        max_tokens_per_window=max_tokens_per_window,
        num_runs_per_function=num_runs_per_function,
        overlap_tokens=overlap_tokens,
        output_dir=output_dir,
    )

    # Report success or failure
    if knowledge_graph:
        num_entities = len(knowledge_graph.get("nodes", {}))
        num_relations = len(knowledge_graph.get("edges", []))
        print(
            f"\nSuccessfully extracted knowledge graph with {num_entities} entities and {num_relations} relations"
        )
        return 0
    else:
        print("\nFailed to extract knowledge graph")
        return 1


ingest_kg_from_script()
