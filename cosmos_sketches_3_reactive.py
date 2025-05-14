import os, json, tenacity
from pydantic import BaseModel, ValidationError, RootModel
from src.vectors.cosmos_client import SimpleCosmosClient
from src.llms.basic_agent import BasicAgent

COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"  # Replace with your database name if different
CONTAINER_NAME = "knowledge-chunks"  # Replace with your container name if different
PARTITION_KEY_PATH = "/id"  # Replace with your partition key path if different

cosmos = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)
cosmos.connect()
last_notes_str=str(cosmos.get_notes_from_day(cosmos.get_last_newsletter_date()))

base_prompt = f"""This is a content of last newslettters about AI: {last_notes_str} ### 
Identify all LLMs (both general names and specific models) mentioned in the text and tell what is the news or opinions about about it.
Present it in json format, where major keys are LLM companies (e.g. OpenAI, Anthropic, Google), then sub key are LLMs version and 
final values are lists of dictionaries with keys: model_name, news, opinion. Use the following example as a template:

{{
    "Google": {{
        "Gemini 2.0": [
            {{"news": "Gemini 2.0 has been released with new features."}},
            {{"opinion": "This is a competitive move against other LLMs."}}
        ],
        "Gemini 2.5 Pro": [
            {{"news": ""Gemini 2.5 Pro's video understanding capabilities are highlighted, noting that...."}},
            {{"opinion": "This is a competitive move against other LLMs."}}
        ]            
    }},
    "Anthropic": {{
        "Claude 3": [
            {{"news": "Claude 3 has been released with new features."}},
            {{"opinion": "This is a competitive move against other LLMs"}}
        ],
        "Claude Code: [
            {{"news": "Claude 3 has been released with new features."}},
            {{"opinion": "This is a competitive move against other LLMs"}}
        ],            
    }}
}}

"""

class Item(BaseModel): 
    model_name:str; 
    news:str; 
    opinion:str

class Payload(RootModel):
    root: dict[str, dict[str, list[Item]]]

agent=BasicAgent()

def query_llm(prompt,max_tries=3,model="gemini-2.0-flash-exp"):
    p=prompt
    for _ in range(max_tries):
        txt=agent.get_text_response_from_llm(model,messages=p,code_tag=None)["text_response"]
        if "```" in txt:
            parts = txt.split("```")
            if len(parts) >= 3: 
                txt = parts[1]
                if txt.startswith("json"):
                    txt = txt[4:].strip()
        try:
            data=json.loads(txt)
            validated_data = Payload.model_validate(data)
            return validated_data.root
        except (json.JSONDecodeError,ValidationError) as err:
            p=f"{txt}\n\nYour JSON was invalid ({err}). Fix it and return only valid JSON."
    raise RuntimeError("Failed to obtain valid payload after 3 attempts.")

result=query_llm(base_prompt)
print(result)
