# import json
import re
import typing
import yaml
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from llms.llm_clients import LLMClientActivator, get_model_name_and_location

def translate_messages_from_openai_to_gemini(
    messages_to_change: list[dict[str, str]],
) -> str:
    last_message = messages_to_change[-1]["content"]
    if len(messages_to_change) == 1:
        gemini_messages = []
    else:
        prev_messages = messages_to_change[:-1]
        gemini_messages = []
        for message in prev_messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                role = "model"

            gemini_messages.append({"role": role, "parts": [content]})

    return gemini_messages, last_message

class BasicAgent:
    def __init__(self):

        with open(os.path.join("llms", "llm_config.yaml"), "r") as file:
            config = yaml.safe_load(file)

        self.llm_model_dict = config["llm_location"]

    def set_llm_client(self, llm_model_name: str):
        self.llm_client = LLMClientActivator(
            llm_model_dict=self.llm_model_dict,
        ).activate_llm_client(llm_model_name)
        return self.llm_client

    @retry(
        wait=wait_fixed(2) + wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def get_text_response_from_llm(
        self,
        llm_model_name: str,
        messages: typing.Union[str, list[dict[str, str]]],
        code_tag: str = None,
    ) -> dict: # Adjusted return type hint to dict based on observed return value
        # Check if messages is a string and wrap it if necessary
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        llm_client = self.set_llm_client(llm_model_name)
        model_location, llm_model_name = get_model_name_and_location(
            self.llm_model_dict, llm_model_name
        )


        reasoning_content = None
        text_content = None

        if model_location in [
            "azure_openai",
            "dbrx",
            "groq",
            "openrouter",
            "priv_openai",
            "deepseek",
        ]:

            my_response = llm_client.chat.completions.create(
                model=llm_model_name,
                messages=messages,
            )

            text_content = my_response.choices[0].message.content
            if "reasoning_content" in my_response.choices[0].message:
                reasoning_content = my_response.choices[0].message.reasoning_content

        elif model_location in ["google_ai_studio"]:
            gemini_messages, last_message = translate_messages_from_openai_to_gemini(
                messages
            )
            chat_session = llm_client.start_chat(history=gemini_messages)
            response = chat_session.send_message(last_message)
            text_content = response.text



        if code_tag is None:
            return {
                "text_response": text_content,
                "reasoning_content": reasoning_content,
            }
        else:
            tool_escaping_pattern = rf"```\s?{code_tag}\s?(.*?)```"
            match = re.search(tool_escaping_pattern, text_content, re.DOTALL)
            if match:
                my_match = match.group(1).strip()
                return {"text_response": my_match}
