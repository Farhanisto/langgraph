#!/Users/farhan/codebase/langgraph/venv/bin/python

import sys
print("Running Python from:", sys.executable)

from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicState):
    print("DEBUG STATE:", state)
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(BasicState)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

print("LangGraph chatbot ready. Type 'exit' to stop.")

while True:
    user_input = input("user: ")
    if user_input.lower() in ["exit", "end"]:
        break
    else:
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            print("AI:", result["messages"][-1].content)
        except Exception as e:
            print("Error:", e)


