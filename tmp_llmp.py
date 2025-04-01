from llms.basic_agent import BasicAgent

llm_agent = BasicAgent()
sth = llm_agent.get_text_response_from_llm(
    llm_model_name="gemini-2.0-flash-exp",
    messages="do you know neo4j?",
    code_tag=None,
)
print(sth)