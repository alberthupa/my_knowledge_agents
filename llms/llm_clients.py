from dotenv import load_dotenv
import os
from openai import AzureOpenAI, OpenAI
from groq import Groq
import google.generativeai as genai

load_dotenv()

def get_model_name_and_location(llm_model_dict: dict, llm_model: str) -> str:
    if ":" in llm_model:
        model_location = llm_model.split(":")[0]
        model_name = llm_model.split(":")[1]
    else:
        llm_locations = [
            key for key, models in llm_model_dict.items() if llm_model in models
        ]
        model_location = llm_locations[0]
        model_name = llm_model

    return model_location, model_name


class LLMClientActivator:
    def __init__(
        self,
        llm_model_dict: dict,
        # credentials_config: None,
    ):
        # self.credentials_config = credentials_config
        self.llm_model_dict = llm_model_dict

    def activate_llm_client(self, llm_model: str):
        model_location, llm_model_name = get_model_name_and_location(
            self.llm_model_dict, llm_model
        )
        print(f"model_location: {model_location} llm_model_name: {llm_model_name}")

        if model_location == "azure_openai":
            return AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
            )
        elif model_location == "priv_openai":
            return OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif model_location == "dbrx":
            return OpenAI(
                api_key=os.environ.get("DATABRICKS_TOKEN"),
                base_url=os.environ.get("DATABRICKS_ENDPOINT"),
                temperature=0,
            )
        elif model_location == "openrouter":
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        elif model_location == "deepseek":
            return OpenAI(
                base_url="https://api.deepseek.com/",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )

        elif model_location == "groq":
            return Groq(api_key=os.environ.get("GROQ_API_KEY"))
        elif model_location == "google_ai_studio":
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            generation_config = {
                "temperature": 0,
                "response_mime_type": "text/plain",
                "max_output_tokens": 8192,
            }
            return genai.GenerativeModel(
                model_name=llm_model,
                generation_config=generation_config,
            )
