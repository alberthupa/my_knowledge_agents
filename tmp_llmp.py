from llms.basic_agent import BasicAgent

llm_agent = BasicAgent()
ai_response = llm_agent.get_text_response_from_llm(
    llm_model_input="gemini-2.0-flash-exp",  # Changed keyword argument
    messages="hi",
    code_tag=None,
)
ai_text_response = ai_response["text"]
