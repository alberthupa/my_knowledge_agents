# works, but ontology works on full file

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


# --- Load Prompt Templates ---
def load_prompt_template(template_file):
    """Load a prompt template from a file."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompts", template_file
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt template '{template_file}': {e}")
        # Return a basic template as fallback
        return "ERROR LOADING TEMPLATE: {0}"


# --- Knowledge Graph Extraction Per Window ---
def extract_window_relations_and_attributes(
    llm_agent,
    llm_model: str,
    file_path: str,
    window_text: str,
    window_num: int,
    total_windows: int,
    entities: set,
    encoding: tiktoken.Encoding,
    max_tokens: int,
    current_knowledge_graph: dict = None,
):
    """
    Extract relations between entities and attributes from a single text window
    to incrementally build a knowledge graph.

    Args:
        llm_agent: The LLM agent to use for extraction
        llm_model: The LLM model name
        file_path: Path to the file being processed
        window_text: Text of the current window
        window_num: Current window number
        total_windows: Total number of windows
        entities: Set of entities found in the document
        encoding: Tiktoken encoding
        max_tokens: Maximum tokens for LLM context
        current_knowledge_graph: Existing knowledge graph to build upon

    Returns:
        dict: Updated knowledge graph with nodes (entities+attributes) and edges (relations)
    """
    # Initialize or use existing knowledge graph
    if current_knowledge_graph is None:
        knowledge_graph = {
            "nodes": {},  # Will store entities and their attributes
            "edges": [],  # Will store relations between entities
        }
        # Create initial nodes for all entities
        for entity in entities:
            knowledge_graph["nodes"][entity] = {"attributes": {}}
    else:
        knowledge_graph = current_knowledge_graph

    # If we have no entities, return empty graph
    if not entities:
        return knowledge_graph

    # Get window token count
    window_token_count = num_tokens_from_string(window_text, encoding)
    window_preview = window_text[:50] + "..." if len(window_text) > 50 else window_text

    # Calculate how much of the text we can include for context
    entity_list = ", ".join(sorted(list(entities)))

    # Describe the current state of the knowledge graph
    existing_relations = []
    for edge in knowledge_graph["edges"]:
        existing_relations.append(
            f"{edge['source']} {edge['relation']} {edge['target']}"
        )

    existing_relations_text = "None"
    if existing_relations:
        existing_relations_text = "\n".join(existing_relations)

    # Load and format the prompt template
    prompt_template = load_prompt_template("relations_prompt.txt")
    prompt_template = prompt_template.format(
        window_num=window_num,
        total_windows=total_windows,
        file_name=os.path.basename(file_path),
        entity_list=entity_list,
        existing_relations_text=existing_relations_text,
    )

    template_tokens = num_tokens_from_string(prompt_template, encoding)
    available_tokens = (
        max_tokens - template_tokens - 100
    )  # Reserve 100 tokens for response

    if available_tokens <= 0:
        print(f"  - Error: Prompt template too long for max_tokens={max_tokens}")
        return knowledge_graph

    # Truncate text to fit available tokens
    text_tokens = encoding.encode(window_text)
    if len(text_tokens) > available_tokens:
        text_tokens = text_tokens[:available_tokens]
        window_text = encoding.decode(text_tokens)

    # Create the full prompt
    prompt = prompt_template + window_text + "\n---"

    print(
        f'  - Extracting relations and attributes for window {window_num}/{total_windows} ({window_token_count} tokens): "{window_preview}"'
    )
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
                                # Merge new attributes with existing ones
                                if "attributes" not in knowledge_graph["nodes"][entity]:
                                    knowledge_graph["nodes"][entity]["attributes"] = {}
                                knowledge_graph["nodes"][entity]["attributes"].update(
                                    data["attributes"]
                                )

                # Add relations to knowledge graph
                new_edges = 0
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
                                # Check if this relation is already in the graph
                                relation_exists = False
                                for edge in knowledge_graph["edges"]:
                                    if (
                                        edge["source"] == relation["source"]
                                        and edge["target"] == relation["target"]
                                        and edge["relation"] == relation["relation"]
                                    ):
                                        relation_exists = True
                                        break

                                if not relation_exists:
                                    knowledge_graph["edges"].append(
                                        {
                                            "source": relation["source"],
                                            "relation": relation["relation"],
                                            "target": relation["target"],
                                        }
                                    )
                                    new_edges += 1

                print(
                    f"  - Window {window_num}: Added {new_edges} new relations, knowledge graph now has {len(knowledge_graph['edges'])} relations total"
                )
            else:
                print(
                    f"  - Failed to locate JSON in LLM response for window {window_num}"
                )
        except json.JSONDecodeError as e:
            print(
                f"  - Failed to parse knowledge graph JSON for window {window_num}: {e}"
            )
            print(f"  - Response was: {response_text[:200]}...")
    except Exception as e:
        print(f"  - Error extracting relations for window {window_num}: {e}")

    return knowledge_graph


# --- Entity Extraction From Windows ---
def extract_entities_from_windows(
    llm_agent, llm_model: str, file_path: str, windows: list
):
    """
    Extract entities from a list of text windows.

    Args:
        llm_agent: The LLM agent to use for extraction
        llm_model: The LLM model name
        file_path: Path to the file being processed
        windows: List of text windows to process

    Returns:
        set: Set of unique entities found in the document
    """
    current_file_entities = set()  # Initialize entities for this file
    encoding = tiktoken.get_encoding(ENCODING_NAME)  # Get encoding for token counting

    print(f"  - [PHASE 1] Starting entity extraction...")

    # Load the entities prompt template
    entities_prompt_template = load_prompt_template("entities_prompt.txt")

    for i, window in enumerate(windows):
        # Get window token count and preview
        window_token_count = num_tokens_from_string(window, encoding)
        window_preview = window[:50] + "..." if len(window) > 50 else window

        # Format the prompt template
        prompt = entities_prompt_template.format(
            file_name=os.path.basename(file_path),
            current_entities=sorted(list(current_file_entities)),
            window_text=window,
        )

        print(
            f'    - Processing window {i+1}/{len(windows)} for entities ({window_token_count} tokens): "{window_preview}"'
        )
        try:
            ai_response = llm_agent.get_text_response_from_llm(
                llm_model_input=llm_model,
                messages=prompt,
            )
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
                print(f"      - Found new entities in window {i+1}: {new_entities}")
                current_file_entities.update(new_entities)

    print(f"  - [PHASE 1] Finished entity extraction.")
    print(f"  - Total unique entities found: {len(current_file_entities)}")

    if current_file_entities:
        print(f"  - Entities: {sorted(list(current_file_entities))}")

    return current_file_entities


# --- Process Single File Function ---
def process_file(
    llm_model: str,
    secondary_llm_model: str,
    file_path: str,
    T: int,
    N: int,
    O: int,
):
    """
    Process a single file, extract entities and build a knowledge graph.

    Args:
        llm_model (str): The primary LLM model to use for extraction.
        secondary_llm_model (str): The secondary LLM model for the second run (if N=2).
        file_path (str): Path to the file to process.
        T (int): Maximum tokens per window.
        N (int): Number of runs for each function per file (1 or 2).
        O (int): Overlap tokens between windows.

    Returns:
        dict: The knowledge graph for the file or None if processing failed
    """
    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
    except Exception as e:
        print(f"Error getting tiktoken encoding '{ENCODING_NAME}': {e}")
        print("Please ensure 'tiktoken' is installed.")
        return None

    llm_agent = BasicAgent()  # Instantiate the agent
    final_knowledge_graph = None

    print(f"\nProcessing file: {file_path}")
    print(f"  - Primary LLM: {llm_model}")
    if N == 2:
        if not secondary_llm_model:
            print(
                "  - Warning: N=2 but secondary_llm_model not provided. Skipping second run."
            )
            N = 1  # Fallback to single run
        else:
            print(f"  - Secondary LLM: {secondary_llm_model}")
    try:
        text = extract_text(file_path)
        if not text:
            print(f"  - No text extracted or file empty: {file_path}")
            return None

        # Generate windows once per file
        windows = list(generate_text_windows(text, encoding, T, O))
        if not windows:
            print(
                f"  - No windows generated (text might be shorter than T): {file_path}"
            )
            return None

        print(f"  - Generated {len(windows)} windows (T={T}, O={O})")

        # --- PHASE 1: Entity Extraction ---
        print(
            f"  - [PHASE 1] Starting entity extraction (Run 1 - Model: {llm_model})..."
        )
        entities_run1 = extract_entities_from_windows(
            llm_agent=llm_agent,
            llm_model=llm_model,
            file_path=file_path,
            windows=windows,
        )

        final_entities = entities_run1

        if N == 2 and secondary_llm_model:
            print(
                f"  - [PHASE 1] Starting entity extraction (Run 2 - Model: {secondary_llm_model})..."
            )
            entities_run2 = extract_entities_from_windows(
                llm_agent=llm_agent,
                llm_model=secondary_llm_model,
                file_path=file_path,
                windows=windows,
            )
            final_entities = entities_run1.union(entities_run2)
            print(
                f"  - [PHASE 1] Combined unique entities from both runs: {len(final_entities)}"
            )
            if final_entities:
                print(f"  - Combined Entities: {sorted(list(final_entities))}")

        if not final_entities:
            print(
                f"  - No entities found after all runs, skipping knowledge graph extraction"
            )
            return None

        # --- PHASE 2: Incremental Knowledge Graph Extraction ---
        print(
            f"  - [PHASE 2] Starting incremental knowledge graph extraction (Run 1 - Model: {llm_model})..."
        )
        kg_run1 = None  # Initialize empty knowledge graph for run 1

        for i, window in enumerate(windows):
            kg_run1 = extract_window_relations_and_attributes(
                llm_agent=llm_agent,
                llm_model=llm_model,
                file_path=file_path,
                window_text=window,
                window_num=i + 1,
                total_windows=len(windows),
                entities=final_entities,  # Use combined entities
                encoding=encoding,
                max_tokens=T,
                current_knowledge_graph=kg_run1,
            )

        final_knowledge_graph = kg_run1  # Start with the result of run 1

        if N == 2 and secondary_llm_model:
            print(
                f"  - [PHASE 2] Starting incremental knowledge graph extraction (Run 2 - Model: {secondary_llm_model})..."
            )
            # Run 2 refines the graph from Run 1
            kg_run2 = kg_run1  # Start Run 2 with the graph from Run 1
            for i, window in enumerate(windows):
                kg_run2 = extract_window_relations_and_attributes(
                    llm_agent=llm_agent,
                    llm_model=secondary_llm_model,
                    file_path=file_path,
                    window_text=window,
                    window_num=i + 1,  # Still use original window numbers for context
                    total_windows=len(windows),
                    entities=final_entities,  # Use combined entities
                    encoding=encoding,
                    max_tokens=T,
                    current_knowledge_graph=kg_run2,  # Pass the evolving graph from this run
                )
            final_knowledge_graph = kg_run2  # The final graph is the result of run 2

        # Print summary of the final knowledge graph
        if final_knowledge_graph:
            num_entities = len(final_knowledge_graph.get("nodes", {}))
            num_relations = len(final_knowledge_graph.get("edges", []))
            print(
                f"  - [PHASE 2] Final knowledge graph completed with {num_entities} nodes and {num_relations} relations"
            )

        return final_knowledge_graph

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Parameters ---
    llm_model = "gemini-2.0-flash"  # Primary model
    secondary_llm_model = "gpt-4o"  # Secondary model for N=2
    # llm_model = "gpt-4o"
    source_directory = "sources"  # Directory containing files
    max_tokens_per_window = 4000  # T: Max tokens in a window
    num_runs_per_function = 2  # N: Set to 1 or 2
    overlap_tokens = 50  # O: Token overlap between windows

    print("--- Starting File Iterator ---")
    print(
        f"Parameters: Dir='{source_directory}', T={max_tokens_per_window}, N={num_runs_per_function}, O={overlap_tokens}"
    )

    # Check if prompts directory exists
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    if not os.path.exists(prompts_dir):
        print(f"Creating prompts directory: {prompts_dir}")
        os.makedirs(prompts_dir, exist_ok=True)

    # Get the first file in the directory
    first_file = None
    try:
        for entry in os.scandir(source_directory):
            if entry.is_file() and entry.name.lower().endswith(SUPPORTED_EXTENSIONS):
                first_file = entry.path
                print(f"Selected file: {first_file}")
                break

        if not first_file:
            print(f"No supported files found in directory: {source_directory}")
            exit(1)
    except FileNotFoundError:
        print(f"Error: Directory not found: {source_directory}")
        exit(1)
    except Exception as e:
        print(f"Error scanning directory {source_directory}: {e}")
        exit(1)

    # Process the single file
    knowledge_graph = process_file(
        llm_model=llm_model,
        secondary_llm_model=secondary_llm_model,
        file_path=first_file,
        T=max_tokens_per_window,
        N=num_runs_per_function,
        O=overlap_tokens,
    )

    # Save the knowledge graph if it was created
    if knowledge_graph:
        output_dir = "knowledge_graphs"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(first_file)
        output_file = os.path.join(output_dir, f"{base_name}.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(knowledge_graph, f, indent=2)
            print(f"Saved knowledge graph for {base_name} to {output_file}")
        except Exception as e:
            print(f"Error saving knowledge graph for {base_name}: {e}")

    print("--- File Iterator Finished ---")
