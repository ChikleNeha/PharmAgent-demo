import json
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. Define the State
class AgentState(TypedDict):
    # Using Annotated with a 'reducer' like the JS version
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# 2. Define the Tools
@tool
def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
tool_node = ToolNode(tools)

# 3. Initialize Ollama
# Make sure you've run 'ollama pull llama3' in your terminal
model = ChatOllama(
    model="llama3.2", 
    temperature=0
).bind_tools(tools)

# 4. Define Logic Nodes
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    # Check if the model wants to call a tool
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 6. Execute
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": "42"}}
)

print(final_state["messages"][-1].content)