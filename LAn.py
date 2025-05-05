from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# Define the state structure for the graph
class AgentState(TypedDict):
    messages: list


# Placeholder node functions
def agent_step_1(state: AgentState) -> AgentState:
    print("--- Agent Step 1 ---")
    # Example: add a message or modify state here
    return state


def agent_step_2(state: AgentState) -> AgentState:
    print("--- Agent Step 2 ---")
    # Example: add a message or modify state here
    return state


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("agent_step_1", agent_step_1)
builder.add_node("agent_step_2", agent_step_2)
builder.add_edge(START, "agent_step_1")
builder.add_edge("agent_step_1", "agent_step_2")
builder.add_edge("agent_step_2", END)

graph = builder.compile()

if __name__ == "__main__":
    # Example initial state
    initial_state = {"messages": ["Hello, LangGraph!"]}
    for output in graph.stream(initial_state):
        print("Graph output:", output)
