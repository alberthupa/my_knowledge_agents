from llms.basic_agent import BasicAgent

llm_agent = BasicAgent()
sth = llm_agent.get_text_response_from_llm(
    llm_model_input="gemini-2.0-flash-exp", # Changed keyword argument
    #messages="hi",
    messages=[{"role": "user", "content": "hi"}],
    code_tag=None,
)
print(sth)
