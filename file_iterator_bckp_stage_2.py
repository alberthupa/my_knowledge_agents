import os
import tiktoken
from pypdf import PdfReader  # Use pypdf instead of PyPDF2 if installing fresh
import io
from llms.basic_agent import BasicAgent  # Added import
import json

# --- Constants ---
ENCODING_NAME = "o200k_base"
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")


# --- Token Counting ---
def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    """Returns the number of tokens in a text string."""
    # No need to get encoding every time, pass it in
    num_tokens = len(encoding.encode(string))
    return num_tokens


# --- Text Extraction ---
def extract_text(file_path: str) -> str:
    """Extracts text from txt, md, or pdf files."""
    _, extension = os.path.splitext(file_path)
    text = ""
    try:
        if extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif extension == ".pdf":
            reader = PdfReader(file_path)
            string_io = io.StringIO()
            for page in reader.pages:
                string_io.write(page.extract_text() or "")  # Handle None return
            text = string_io.getvalue()
        else:
            print(f"Warning: Unsupported file type skipped: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return text


# --- Window Generation ---
def generate_text_windows(
    text: str, encoding: tiktoken.Encoding, max_tokens: int, overlap_tokens: int
):
    """Generates overlapping text windows based on token count."""
    if not text:
        return

    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    start = 0
    step = max_tokens - overlap_tokens
    if step <= 0:
        print("Warning: Overlap is >= max_tokens. Setting step to 1.")
        step = 1  # Avoid infinite loop or no progress

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        window_tokens = tokens[start:end]
        # Decode the tokens back to a string for the window
        window_text = encoding.decode(window_tokens)
        yield window_text
        if end == total_tokens:
            break  # Reached the end
        start += step


# --- Knowledge Graph Extraction ---
def extract_relations_and_attributes(
    llm_agent,
    llm_model: str,
    file_path: str,
    text: str,
    entities: set,
    encoding: tiktoken.Encoding,
    max_tokens: int,
):
    """
    Extract relations between entities and attributes of entities to build a knowledge graph.

    Args:
        llm_agent: The LLM agent to use for extraction
        llm_model: The LLM model name
        file_path: Path to the file being processed
        text: Full text of the document
        entities: Set of entities found in the document
        encoding: Tiktoken encoding
        max_tokens: Maximum tokens for LLM context

    Returns:
        dict: Knowledge graph with nodes (entities+attributes) and edges (relations)
    """
    # Initialize knowledge graph structure
    knowledge_graph = {
        "nodes": {},  # Will store entities and their attributes
        "edges": [],  # Will store relations between entities
    }

    # Create initial nodes for all entities
    for entity in entities:
        knowledge_graph["nodes"][entity] = {"attributes": {}}

    # If we have no entities, return empty graph
    if not entities:
        return knowledge_graph

    # Calculate how much of the text we can include for context
    entity_list = ", ".join(sorted(list(entities)))
    # Calculate tokens needed for prompt template without the text
    prompt_template = f"""Analyze the text from '{os.path.basename(file_path)}' to extract:
1. ATTRIBUTES: Key properties/characteristics of each entity
2. RELATIONS: How entities are connected to each other

Entities found in document: {entity_list}

For each entity, identify important attributes like definitions, categories, properties, etc.
For relations, specify which entities are connected and how (e.g., "Entity A is part of Entity B", "Entity C depends on Entity D").

FORMAT YOUR RESPONSE AS JSON:
{{
  "entities": {{
    "Entity1": {{
      "attributes": {{"attribute1": "value1", "attribute2": "value2"}}
    }},
    "Entity2": {{
      "attributes": {{"attribute1": "value1", "attribute2": "value2"}}
    }}
  }},
  "relations": [
    {{"source": "Entity1", "relation": "relates to", "target": "Entity2"}},
    {{"source": "Entity2", "relation": "depends on", "target": "Entity3"}}
  ]
}}

Text to analyze:
---
"""

    template_tokens = num_tokens_from_string(prompt_template, encoding)
    available_tokens = (
        max_tokens - template_tokens - 100
    )  # Reserve 100 tokens for response

    if available_tokens <= 0:
        print(f"  - Error: Prompt template too long for max_tokens={max_tokens}")
        return knowledge_graph

    # Truncate text to fit available tokens
    text_tokens = encoding.encode(text)
    if len(text_tokens) > available_tokens:
        text_tokens = text_tokens[:available_tokens]
        text = encoding.decode(text_tokens)

    # Create the full prompt
    prompt = prompt_template + text + "\n---"

    print(f"  - Extracting relations and attributes for knowledge graph...")
    try:
        ai_response = llm_agent.get_text_response_from_llm(
            llm_model_input=llm_model,
            messages=prompt,
        )
        response_text = ai_response.get("text_response", "").strip()

        # Try to parse JSON response
        try:
            # Find JSON content (it might be surrounded by explanatory text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                kg_data = json.loads(json_content)

                # Update knowledge graph with attributes
                if "entities" in kg_data:
                    for entity, data in kg_data["entities"].items():
                        if entity in knowledge_graph["nodes"]:
                            if "attributes" in data:
                                knowledge_graph["nodes"][entity]["attributes"] = data[
                                    "attributes"
                                ]

                # Add relations to knowledge graph
                if "relations" in kg_data:
                    for relation in kg_data["relations"]:
                        if (
                            "source" in relation
                            and "target" in relation
                            and "relation" in relation
                        ):
                            # Only add relation if both source and target entities exist
                            if (
                                relation["source"] in knowledge_graph["nodes"]
                                and relation["target"] in knowledge_graph["nodes"]
                            ):
                                knowledge_graph["edges"].append(
                                    {
                                        "source": relation["source"],
                                        "relation": relation["relation"],
                                        "target": relation["target"],
                                    }
                                )

                print(
                    f"  - Successfully extracted {len(knowledge_graph['edges'])} relations and attributes for {len(knowledge_graph['nodes'])} entities"
                )
            else:
                print(f"  - Failed to locate JSON in LLM response")
        except json.JSONDecodeError as e:
            print(f"  - Failed to parse knowledge graph JSON: {e}")
            print(f"  - Response was: {response_text[:200]}...")
    except Exception as e:
        print(f"  - Error extracting relations: {e}")

    return knowledge_graph


# --- Main Iterator Function ---
def process_files(
    llm_model: str, directory: str, T: int, N: int, O: int, F: int = None
):
    """
    Iterates through files, extracts text, generates windows, and builds a knowledge graph.

    Args:
        llm_model (str): The LLM model to use for extraction.
        directory (str): Path to the directory containing source files.
        T (int): Maximum tokens per window.
        N (int): Number of runs for each dummy function per file.
        O (int): Overlap tokens between windows.
        F (int, optional): Maximum number of files to process. Defaults to None (process all).

    Returns:
        dict: A dictionary mapping file paths to their knowledge graphs
    """
    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
    except Exception as e:
        print(f"Error getting tiktoken encoding '{ENCODING_NAME}': {e}")
        print("Please ensure 'tiktoken' is installed.")
        return {}

    llm_agent = BasicAgent()  # Instantiate the agent
    all_files_entities = {}  # Store entities for all files
    all_knowledge_graphs = {}  # Store knowledge graphs for all files
    processed_files = set()  # Keep track of processed files
    files_to_process = []

    # --- Collect files ---
    print(f"Scanning directory: {directory}")
    try:
        count = 0
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.lower().endswith(SUPPORTED_EXTENSIONS):
                if F is None or count < F:
                    files_to_process.append(entry.path)
                    count += 1
                else:
                    print(f"Reached file limit F={F}")
                    break  # Stop if file limit F is reached
        print(f"Found {len(files_to_process)} files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return {}
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return {}

    # --- Process files ---
    for file_path in files_to_process:
        if file_path in processed_files:
            # This check might be redundant if we only populate files_to_process once,
            # but good for potential future modifications.
            print(
                f"Skipping already processed file (should not happen with current logic): {file_path}"
            )
            continue

        print(f"\nProcessing file: {file_path}")
        try:
            text = extract_text(file_path)
            if not text:
                print(f"  - No text extracted or file empty: {file_path}")
                processed_files.add(file_path)  # Mark as processed even if empty/error
                continue

            # Generate windows once per file
            windows = list(generate_text_windows(text, encoding, T, O))
            if not windows:
                print(
                    f"  - No windows generated (text might be shorter than T): {file_path}"
                )
                processed_files.add(file_path)
                continue

            print(f"  - Generated {len(windows)} windows (T={T}, O={O})")

            # --- Entity Extraction using LLM ---
            print(f"  - Starting entity extraction...")
            current_file_entities = set()  # Initialize entities for this file

            for i, window in enumerate(windows):
                prompt = f"""Context: The following entities have been extracted from the document '{os.path.basename(file_path)}' so far: {sorted(list(current_file_entities))}

Task: Analyze the text window below, asnwer question: what is this text about? What are the most imporant entities (key concepts) in it? 
Do not include persons or organizations.
Present entities ONLY in this window that are NOT already listed in the context above. List the new entities found in this window, one per line. If no new entities are found in this specific window, respond with the single word "None".

Text Window:
---
{window}
---

New Entities (only from this window):"""

                print(
                    f"    - Processing window {i+1}/{len(windows)} for entities..."
                )  # Add progress indicator
                try:
                    ai_response = llm_agent.get_text_response_from_llm(
                        llm_model_input=llm_model,  # Use model from example
                        messages=prompt,
                        # code_tag is omitted as per example
                    )
                    # Safely get the text response, default to empty string if key missing or response is None
                    ai_text_response = ai_response.get("text_response", "").strip()
                except Exception as e:
                    print(f"      - Error calling LLM for window {i+1}: {e}")
                    ai_text_response = ""  # Treat as no response on error

                if ai_text_response and ai_text_response.lower() != "none":
                    # Split by newline, strip whitespace, filter out empty lines
                    new_entities = {
                        entity.strip()
                        for entity in ai_text_response.split("\n")
                        if entity.strip()
                    }
                    if new_entities:
                        print(
                            f"      - Found new entities in window {i+1}: {new_entities}"
                        )
                        current_file_entities.update(new_entities)

            # Store entities for the file
            all_files_entities[file_path] = current_file_entities
            print(f"  - Finished entity extraction for {file_path}.")
            print(f"  - Total unique entities found: {len(current_file_entities)}")
            if current_file_entities:  # Print entities if any were found
                print(f"  - Entities: {sorted(list(current_file_entities))}")

                # --- Knowledge Graph Extraction ---
                print(f"  - Starting knowledge graph extraction...")
                knowledge_graph = extract_relations_and_attributes(
                    llm_agent,
                    llm_model,
                    file_path,
                    text,
                    current_file_entities,
                    encoding,
                    T * 2,  # Use larger context for knowledge graph extraction
                )

                # Store the knowledge graph
                all_knowledge_graphs[file_path] = knowledge_graph

                # Print summary of knowledge graph
                num_entities = len(knowledge_graph["nodes"])
                num_relations = len(knowledge_graph["edges"])
                print(
                    f"  - Knowledge graph created with {num_entities} nodes and {num_relations} relations"
                )

            processed_files.add(file_path)  # Mark as successfully processed

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(
        f"\nFinished processing all files. Total files processed: {len(processed_files)}"
    )
    print(f"Total knowledge graphs created: {len(all_knowledge_graphs)}")

    return all_knowledge_graphs


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Parameters ---
    llm_model = "gemini-2.0-flash"
    source_directory = "sources"  # Directory containing files
    max_tokens_per_window = 4000  # T: Max tokens in a window
    num_runs_per_function = 1  # N: Number of times to run each approach
    overlap_tokens = 50  # O: Token overlap between windows
    max_files_to_process = None  # F: Max number of files (None for all)

    print("--- Starting File Iterator ---")
    print(
        f"Parameters: Dir='{source_directory}', T={max_tokens_per_window}, N={num_runs_per_function}, O={overlap_tokens}, F={max_files_to_process}"
    )

    knowledge_graphs = process_files(
        llm_model=llm_model,
        directory=source_directory,
        T=max_tokens_per_window,
        N=num_runs_per_function,
        O=overlap_tokens,
        F=max_files_to_process,
    )

    # Optional: Save knowledge graphs to file
    output_dir = "knowledge_graphs"
    os.makedirs(output_dir, exist_ok=True)

    for file_path, kg in knowledge_graphs.items():
        base_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"{base_name}.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(kg, f, indent=2)
            print(f"Saved knowledge graph for {base_name} to {output_file}")
        except Exception as e:
            print(f"Error saving knowledge graph for {base_name}: {e}")

    print("--- File Iterator Finished ---")
