import os
import argparse
import re
from datetime import datetime
from azure.cosmos import exceptions
from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import Dict, List
from src.vectors.cosmos_client import SimpleCosmosClient
from src.llms.basic_agent import BasicAgent
import json

"""
https://chatgpt.com/c/6824f32a-2c70-800a-8080-e55fd2007674
"""


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
CONTAINER_NAME = "knowledge-chunks"
PARTITION_KEY_PATH = "/id"


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()


StrList = List[str]


class Summary(BaseModel):
    news: str
    keywords: StrList = Field(alias="keywords")
    companies: StrList = Field(alias="companies")
    model_name: StrList = Field(alias="model name")
    model_architecture: StrList = Field(alias="model architecture")
    detailed_model_version: StrList = Field(alias="detailed model version")
    ai_tools: StrList = Field(alias="ai tools")
    infrastructure: StrList = Field(alias="infrastucture")
    ml_techniques: StrList = Field(alias="ml techniques")


class Payload(RootModel):
    root: Dict[str, Summary]  # e.g. {"news summary 1": Summary, …}


def query_llm(
    prompt: str, agent, model: str = "gemini-2.0-flash-exp", max_tries: int = 3
) -> Dict[str, Summary]:
    p = prompt
    for _ in range(max_tries):
        txt = agent.get_text_response_from_llm(model, messages=p, code_tag=None)[
            "text_response"
        ]
        if "```" in txt:
            parts = txt.split("```")
            if len(parts) >= 3:
                txt = parts[1]
                if txt.lstrip().startswith("json"):
                    txt = txt[4:].lstrip()
        try:
            data = json.loads(txt)
            validated = Payload.model_validate(data)
            return validated.root
        except (json.JSONDecodeError, ValidationError) as err:
            p = f"{txt}\n\nYour JSON was invalid ({err}). Fix it and return only valid JSON."
    raise RuntimeError("Failed to obtain valid payload after 3 attempts.")


message_template = """This is a content of last newslettters about AI: {text}.
    Extract news and opinions about AI related topics from the text. Accompany it with keywords organized into topics:
    {{
        "... news headline ..." : {{
            "news": "Gemini 2.0 has been released with new features.",
            "keywords": ["", ""], # one of: "LLMs", "AI Agents", "AI Infrastructure", "AI Engineering & Tooling", "AI Applications", "Benchmarks", "Metrics", "AI Companies", "AI Researchers & Engineers", "AI Communities", "AI Events", "AI Governance", "AI Ethics", "AI Bias", "AI Impact on Work", "Copyright", "Datasets", "Model Weights", "Technical Reports", "Open Source Projects",
            "companies": ["", ""], # e.g. "Google", "Anthropic", "OpenAI", "Meta", "Tencent", "DeepSeek", "Perplexity AI", "Cartesia", "PrimeIntellect", "Alibaba", "HuggingFace", "Unsloth AI", "Nous Research AI",
            "model name": ["", ""], #  e.g. "Gemini 2.0", "Claude 3", "Claude Code", "Gemini 2.5 Pro", "Gemini 2.5 Turbo",
            "model architecture": ["", ""], # e.g. "Transformer", "MoE", "Llama", "Claude", "Gemini", "DeepSeek", "Grok", "Sonar",
            "detailed model version": ["", ""], # e.g. "Gemini 2.0", "Claude 3", "Claude Code", "Gemini 2.5 Pro", "Gemini 2.5 Turbo",
            "ai tools: ["", ""], # e.g. "KerasRS", "LangChain", "LlamaIndex", "Aider", "Cursor", "Windsurf", "CUTLASS", "CuTe DSL", "Torchtune", "Mojo",
            "infrastucture": ["", ""], # e.g. "NVIDIA", "Intel Arc", "TPUs", "VRAM", "KV Cache", "CUDA", "IPEX", "SYCL",
            "ml techniques": ["", ""] # e.g. "XGBoost", "RL", "Supervised Learning", "Post-Training", "Quantization Techniques", "Inference Optimization
        }}
    }}   
"""


if cosmos_client.container_client:
    last_date = cosmos_client.get_last_newsletter_date()
    last_notes = cosmos_client.get_notes_from_day(last_date)

    for note in last_notes:
        message = message_template.format(text=note["text"])
        cleaned_payload = query_llm(message, BasicAgent())
        for tag, summary in cleaned_payload.items():
            if len(summary.companies) > 0:
                print(f"Companies: {summary.companies}")
