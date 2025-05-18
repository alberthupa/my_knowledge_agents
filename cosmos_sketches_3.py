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
import hashlib

"""
https://chatgpt.com/c/6824f32a-2c70-800a-8080-e55fd2007674
https://gemini.google.com/app/362b4f461787e386
"""


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
PARTITION_KEY_PATH = "/id"
# CONTAINER_NAME = "knowledge-chunks"


def make_piece_id(parent_id: str, headline: str) -> str:
    digest = hashlib.sha1(headline.encode("utf-8")).hexdigest()[:12]
    return f"{parent_id}_{digest}"


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
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
        print(f"attempt {_ + 1} of {max_tries}")
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


message_template = """This is a content of last newslettters about AI: {text}. #####
    Your task is to extract news and opinions about AI related topics from the text. Accompany it with keywords organized into topics:
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


if cosmos_client:
    done_ids = cosmos_client.run_query(
        container_name="knowledge-pieces", query="SELECT VALUE p.id FROM p"
    )
    pieces = cosmos_client.database_client.get_container_client("knowledge-pieces")
    chunks = cosmos_client.database_client.get_container_client("knowledge-chunks")
    chunks_to_do = chunks.query_items(
        query="""
            SELECT TOP 1 *
            FROM c
            WHERE NOT ARRAY_CONTAINS(@done, c.id)
            ORDER BY c.chunk_date  
        """,
        parameters=[{"name": "@done", "value": list(done_ids)}],
        enable_cross_partition_query=True,
    )
    chunk_to_do = list(chunks_to_do)[0]

    print(f"Chunk to do: {chunk_to_do['id']}")
    # print(chunk_to_do)
    message = message_template.format(text=chunk_to_do["text"])
    cleaned_payload = query_llm(message, BasicAgent())

    keys_to_copy = {
        "id",
        "source",
        "chunk_date",
        "processing_target_date",
    }

    pieces_to_paste = {k: v for k, v in chunk_to_do.items() if k in keys_to_copy}

    for headline, summary in cleaned_payload.items():
        piece_to_paste = pieces_to_paste.copy()  # id, source, dates, …
        piece_to_paste["parent_id"] = chunk_to_do["id"]

        # stable, human‐readable id
        piece_to_paste["id"] = make_piece_id(chunk_to_do["id"], headline)

        piece_to_paste["headline"] = headline
        for k, v in summary.model_dump().items():
            if k != "id":  # avoid clashing with our own id
                piece_to_paste[k] = v

        pieces.upsert_item(
            piece_to_paste,  # idempotent write
            # partition_key=piece_to_paste["id"],
        )


"""
last_date = cosmos_client.get_last_date("knowledge-chunks")
print(f"Last date: {last_date}")
last_notes = cosmos_client.get_notes_from_day("knowledge-chunks", last_date)
if len(last_notes) == 0:
    print("No notes found.")

else:
    for note in last_notes:
        print(note["id"])
        # message = message_template.format(text=note["text"])
        # cleaned_payload = query_llm(message, BasicAgent())
        # for tag, summary in cleaned_payload.items():
        #    print(f"Tag: {tag}")
        #    print(f"News: {summary}")
        #    # if len(summary.companies) > 0:
        #    #    print(f"Companies: {summary.companies}")
"""
