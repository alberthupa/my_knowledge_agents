from dotenv import load_dotenv
import os
from openai import AzureOpenAI, OpenAI
from groq import Groq
import google.generativeai as genai

load_dotenv()

def create_llm_client(llm_model_input: str, llm_model_dict: dict):
    """
    Parses the model input string, determines the location and actual model name,
    and instantiates the appropriate LLM client.

    Args:
        llm_model_input: The model identifier string (e.g., "azure_openai:gpt-4", "gpt-4").
        llm_model_dict: Dictionary mapping locations to lists of supported models.

    Returns:
        A tuple containing:
        - client: The instantiated LLM client object.
        - model_location: The determined location (e.g., "azure_openai").
        - resolved_model_name: The specific model name (e.g., "gpt-4").
        Returns (None, None, None) if configuration is missing or invalid.
    """
    # 1. Determine model location and name
    if ":" in llm_model_input:
        try:
            model_location, resolved_model_name = llm_model_input.split(":", 1)
            if model_location not in llm_model_dict:
                 print(f"Warning: Explicit location '{model_location}' not found in config keys. Proceeding anyway.")
            # Optional: Validate if resolved_model_name is in llm_model_dict[model_location] if needed
        except ValueError:
            print(f"Error: Invalid model input format '{llm_model_input}'. Expected 'location:model_name' or 'model_name'.")
            return None, None, None
    else:
        resolved_model_name = llm_model_input
        # Find the location from the dictionary
        llm_locations = [
            key for key, models in llm_model_dict.items() if resolved_model_name in models
        ]
        if not llm_locations:
            print(f"Error: Model '{resolved_model_name}' not found in any location in the config.")
            return None, None, None
        model_location = llm_locations[0] # Take the first location found

    print(f"Attempting to activate client for: location='{model_location}', model='{resolved_model_name}'")

    # 2. Instantiate the client based on location
    client = None
    try:
        if model_location == "azure_openai":
            client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
            )
        elif model_location == "priv_openai":
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif model_location == "dbrx":
            # Assuming temperature is a default, can be overridden in the call if needed
            client = OpenAI(
                api_key=os.environ.get("DATABRICKS_TOKEN"),
                base_url=os.environ.get("DATABRICKS_ENDPOINT"),
            )
        elif model_location == "openrouter":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        elif model_location == "deepseek":
            client = OpenAI(
                base_url="https://api.deepseek.com/",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
        elif model_location == "groq":
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        elif model_location == "google_ai_studio":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                 raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            # Configuration can be set per-request if needed, simplifying client creation
            generation_config = {
                "temperature": 0, # Default, can be overridden
                "response_mime_type": "text/plain",
                "max_output_tokens": 8192, # Default, can be overridden
            }
            client = genai.GenerativeModel(
                model_name=resolved_model_name,
                generation_config=generation_config, # Apply default config
            )
        else:
            print(f"Error: Unknown model location '{model_location}'.")
            return None, None, None

        if client is None:
             # This might happen if an env var is missing for a specific client type
             print(f"Error: Client creation failed for location '{model_location}'. Check environment variables.")
             return None, None, None

    except Exception as e:
        print(f"Error during client instantiation for {model_location}: {e}")
        return None, None, None

    return client, model_location, resolved_model_name

# Removed LLMClientActivator class and get_model_name_and_location function
