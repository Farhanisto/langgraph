from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")

GENERATE_POST = "genarate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK= "collect_feedback"


def generate_post(state: State)-> State:
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def get_review_decision(state:State):
    generated_post = state["messages"][-1].content
    print("\n review this \n")
    print(generated_post)
    print("\n")
    decision = input("post this? yes or no:")
    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK

def post(state:State):
    final_post = state["messages"][-1]
    print("final linkedIn post \n")
    print(final_post)
    print("\n✅ Post has been approved and is now live on LinkedIn!")

def collect_feedback(state:State):
    feedback = input("How can this be improved")
    return {
        "messages": [HumanMessage(content= feedback)]
    }

graph = StateGraph(State)
graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)
graph.add_node(POST, post)

graph.set_entry_point(GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()
response = app.invoke({
    "messages": [HumanMessage(content="Write me a LinkedIn post on AI Agents taking over content creation")]
})

print(response["messages"][-1].content)
