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


from tqdm import tqdm
import time


"""
https://chatgpt.com/c/6824f32a-2c70-800a-8080-e55fd2007674
https://gemini.google.com/app/362b4f461787e386
"""


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
PARTITION_KEY_PATH = "/id"


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()
pieces = cosmos_client.database_client.get_container_client("knowledge-pieces")
chunks = cosmos_client.database_client.get_container_client("knowledge-chunks")


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
    # prompt: str, agent, model: str = "gemini-2.0-flash-exp", max_tries: int = 3
    prompt: str,
    agent,
    model: str = "gemini-2.0-flash-exp",
    max_tries: int = 3,
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


def make_piece_id(parent_id: str, headline: str) -> str:
    digest = hashlib.sha1(headline.encode("utf-8")).hexdigest()[:12]
    return f"{parent_id}_{digest}"


keys_to_copy = [
    "id",
    "source",
    "chunk_date",
    "processing_target_date",
]

query_to_get_a_note = """
    SELECT TOP 1000 *
    FROM c
    WHERE NOT ARRAY_CONTAINS(@done, c.id)
    ORDER BY c.chunk_date  
"""


def process_chunk(chunk_to_do, done_ids):
    try:
        print(f"Processing chunk: {chunk_to_do['id']}")
        message = message_template.format(text=chunk_to_do["text"])
        cleaned_payload = query_llm(message, BasicAgent(), model="priv_openai:gpt-4.1")
        # cleaned_payload = query_llm(message, BasicAgent(), model="gemini-2.0-flash-exp")
        pieces_to_paste = {k: v for k, v in chunk_to_do.items() if k in keys_to_copy}
        for headline, summary in cleaned_payload.items():
            piece_to_paste = pieces_to_paste.copy()
            piece_to_paste["parent_id"] = chunk_to_do["id"]
            piece_to_paste["id"] = make_piece_id(chunk_to_do["id"], headline)
            piece_to_paste["headline"] = headline
            for k, v in summary.model_dump().items():
                if k != "id":
                    piece_to_paste[k] = v
            pieces.upsert_item(piece_to_paste)
        return True
    except Exception as e:
        print(f"Error processing chunk {chunk_to_do['id']}: {e}")
        return False


if cosmos_client:
    done_ids = pieces.query_items(
        query="SELECT VALUE p.parent_id FROM p",
        enable_cross_partition_query=True,
    )
    list_done_ids = list(done_ids)
    # print(list_done_ids)
    print("----")
    print(f"len done_ids: {len(list_done_ids)}")
    print(type(list_done_ids))

    chunks_to_do = chunks.query_items(
        query=query_to_get_a_note,
        parameters=[{"name": "@done", "value": list(list_done_ids)}],
        enable_cross_partition_query=True,
    )
    chunks_to_do = list(chunks_to_do)

    with tqdm(total=len(chunks_to_do), desc="Processing chunks") as pbar:
        for chunk_to_do in chunks_to_do:
            if process_chunk(chunk_to_do, done_ids):
                pbar.update(1)
            else:
                time.sleep(2)  # Optional: backoff on error

"""

id='codex-mini-latest', created='2025-05-08 05:00:57'
id='gpt-image-1', created='2025-04-24 19:50:30'
id='gpt-4.1-nano', created='2025-04-10 23:48:27'
id='gpt-4.1-nano-2025-04-14', created='2025-04-10 23:37:05'
id='gpt-4.1-mini', created='2025-04-10 22:49:33'
id='gpt-4.1-mini-2025-04-14', created='2025-04-10 22:39:07'
id='gpt-4.1', created='2025-04-10 22:22:22'
id='gpt-4.1-2025-04-14', created='2025-04-10 22:09:06'
id='o4-mini', created='2025-04-09 21:02:31'
id='o4-mini-2025-04-16', created='2025-04-08 19:31:46'
id='gpt-4o-mini-tts', created='2025-03-19 18:05:59'
id='o1-pro', created='2025-03-17 23:49:51'
id='o1-pro-2025-03-19', created='2025-03-17 23:45:04'
id='gpt-4o-mini-transcribe', created='2025-03-15 20:56:36'
id='gpt-4o-transcribe', created='2025-03-15 20:54:23'
id='gpt-4o-mini-search-preview', created='2025-03-08 00:46:01'
id='gpt-4o-mini-search-preview-2025-03-11', created='2025-03-08 00:40:58'
id='gpt-4o-search-preview', created='2025-03-08 00:05:20'
id='gpt-4o-search-preview-2025-03-11', created='2025-03-07 23:56:10'

if cosmos_client:
    done_ids = cosmos_client.run_query(
        container_name="knowledge-pieces", query="SELECT VALUE p.id FROM p"
    )

    chunks_to_do = chunks.query_items(
        query=query_to_get_a_note,
        parameters=[{"name": "@done", "value": list(done_ids)}],
        enable_cross_partition_query=True,
    )
    chunk_to_do = list(chunks_to_do)[0]

    print(f"Chunk to do: {chunk_to_do['id']}")

    message = message_template.format(text=chunk_to_do["text"])
    cleaned_payload = query_llm(message, BasicAgent())
    pieces_to_paste = {k: v for k, v in chunk_to_do.items() if k in keys_to_copy}

    for headline, summary in cleaned_payload.items():
        piece_to_paste = pieces_to_paste.copy()
        piece_to_paste["parent_id"] = chunk_to_do["id"]
        piece_to_paste["id"] = make_piece_id(chunk_to_do["id"], headline)
        piece_to_paste["headline"] = headline
        for k, v in summary.model_dump().items():
            if k != "id":
                piece_to_paste[k] = v

        pieces.upsert_item(
            piece_to_paste,
"""

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
