from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [TavilySearch(max_results=3)]

llm_with_tools = llm.bind_tools(tools=tools)
tool_node = ToolNode(tools=tools)

class BasicChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: BasicChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tools_route(state: BasicChatState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END

graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")
graph.add_edge("tool_node", "chatbot")
graph.add_conditional_edges("chatbot", tools_route)
app = graph.compile()

print("Assistant ready. Type 'exit' to stop.")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "end"]:
        break
    try:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        print("Assistant:", result["messages"][-1].content)
    except Exception as e:
        print("Error:", e)
