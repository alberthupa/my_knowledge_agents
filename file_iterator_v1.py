import os
import tiktoken
from pypdf import PdfReader # Use pypdf instead of PyPDF2 if installing fresh
import io

# --- Constants ---
ENCODING_NAME = "o200k_base"
SUPPORTED_EXTENSIONS = ('.txt', '.md', '.pdf')

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
        if extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif extension == '.pdf':
            reader = PdfReader(file_path)
            string_io = io.StringIO()
            for page in reader.pages:
                string_io.write(page.extract_text() or "") # Handle None return
            text = string_io.getvalue()
        else:
            print(f"Warning: Unsupported file type skipped: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return text

# --- Window Generation ---
def generate_text_windows(text: str, encoding: tiktoken.Encoding, max_tokens: int, overlap_tokens: int):
    """Generates overlapping text windows based on token count."""
    if not text:
        return

    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    start = 0
    step = max_tokens - overlap_tokens
    if step <= 0:
        print("Warning: Overlap is >= max_tokens. Setting step to 1.")
        step = 1 # Avoid infinite loop or no progress

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        window_tokens = tokens[start:end]
        # Decode the tokens back to a string for the window
        window_text = encoding.decode(window_tokens)
        yield window_text
        if end == total_tokens:
            break # Reached the end
        start += step

# --- Dummy Processing Functions ---
def first_approach(text_window: str):
    """Dummy function 1."""
    # Replace with actual logic later
    # print(f"  -> First Approach on window (first 50 chars): {text_window[:50]}...")
    pass

def second_approach(text_window: str):
    """Dummy function 2."""
    # Replace with actual logic later
    # print(f"  -> Second Approach on window (first 50 chars): {text_window[:50]}...")
    pass

# --- Main Iterator Function ---
def process_files(directory: str, T: int, N: int, O: int, F: int = None):
    """
    Iterates through files, extracts text, generates windows, and applies dummy functions.

    Args:
        directory (str): Path to the directory containing source files.
        T (int): Maximum tokens per window.
        N (int): Number of runs for each dummy function per file.
        O (int): Overlap tokens between windows.
        F (int, optional): Maximum number of files to process. Defaults to None (process all).
    """
    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
    except Exception as e:
        print(f"Error getting tiktoken encoding '{ENCODING_NAME}': {e}")
        print("Please ensure 'tiktoken' is installed.")
        return

    processed_files = set() # Dummy index for processed files
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
                    break # Stop if file limit F is reached
        print(f"Found {len(files_to_process)} files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return

    # --- Process files ---
    for file_path in files_to_process:
        if file_path in processed_files:
            # This check might be redundant if we only populate files_to_process once,
            # but good for potential future modifications.
            print(f"Skipping already processed file (should not happen with current logic): {file_path}")
            continue

        print(f"\nProcessing file: {file_path}")
        try:
            text = extract_text(file_path)
            if not text:
                print(f"  - No text extracted or file empty: {file_path}")
                processed_files.add(file_path) # Mark as processed even if empty/error
                continue

            # Generate windows once per file
            windows = list(generate_text_windows(text, encoding, T, O))
            if not windows:
                 print(f"  - No windows generated (text might be shorter than T): {file_path}")
                 processed_files.add(file_path)
                 continue

            print(f"  - Generated {len(windows)} windows (T={T}, O={O})")

            # Apply first_approach N times to all windows
            print(f"  - Applying first_approach {N} times...")
            for run in range(N):
                # print(f"    - Run {run + 1}") # Optional: print run number
                for i, window in enumerate(windows):
                    # print(f"      - Window {i + 1}") # Optional: print window number
                    first_approach(window)

            # Apply second_approach N times to all windows
            print(f"  - Applying second_approach {N} times...")
            for run in range(N):
                # print(f"    - Run {run + 1}")
                for i, window in enumerate(windows):
                    # print(f"      - Window {i + 1}")
                    second_approach(window)

            processed_files.add(file_path) # Mark as successfully processed
            print(f"  - Finished processing {file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Decide if you want to add to processed_files on error or retry later
            # processed_files.add(file_path)

    print(f"\nFinished processing all files. Total processed: {len(processed_files)}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Parameters ---
    source_directory = "sources"  # Directory containing files
    max_tokens_per_window = 500   # T: Max tokens in a window
    num_runs_per_function = 1     # N: Number of times to run each approach
    overlap_tokens = 50           # O: Token overlap between windows
    max_files_to_process = None   # F: Max number of files (None for all)

    print("--- Starting File Iterator ---")
    print(f"Parameters: Dir='{source_directory}', T={max_tokens_per_window}, N={num_runs_per_function}, O={overlap_tokens}, F={max_files_to_process}")

    process_files(
        directory=source_directory,
        T=max_tokens_per_window,
        N=num_runs_per_function,
        O=overlap_tokens,
        F=max_files_to_process
    )

    print("--- File Iterator Finished ---")
