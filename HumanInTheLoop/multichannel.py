from langgraph.graph import add_messages, START, END, StateGraph
from langgraph.types import Command, Interrupt, interrupt
from typing import TypedDict, Annotated, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from IPython.display import display, Image

llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    linkedIn_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

def model(state: State):
    "Here we are using the LLM to generate a linkedIn post with human feedback incorparated"
    print("[model]-Generating agent")
    linkedin_topic = state["linkedIn_topic"]
    # feedback= state.get("human_feedback", "no feedback yet")
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]
    prompt = f"""
    LinkedIn Topic: {linkedin_topic}
    Human Feedback: {feedback[-1] if feedback else "No feedback"}
    Generate a structured and well-written linkedIn Post on the given topic.
    Consider previous human feedback to refine the reponse.
    """
    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer"),
        HumanMessage(content=prompt)
    ])
    generated_ln_post=response.content
    print(f"[model_node] Generated post:\n {generated_ln_post} \n")
    return{
        "generated_post":[AIMessage(content=generated_ln_post)],
        "human_feedback":feedback
    }
def human_node(state:State):
    """Human Intervention node - loops back to model unless input is done"""
    print("\n [human_node] awaiting human feedback...")
    generated_post = state["generated_post"]
    user_feedback = interrupt(
        {
            "generated_post": generated_post, 
            "message": "Provide feedback or type 'done' to finish"
        }
    )

    if user_feedback.lower() == "done":
        return Command(update={"human_feedback":state["human_feedback"] + ["finalised"]}, goto="end_node")
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")
def end_node(state:State):
    """Final Code"""
    print("\n[end_node] Process finished")
    print("Final Generated Post:", state["generated_post"][-1])
    return{"generated_post": state["generated_post"], "human_feedback":state["human_feedback"]}

graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)
graph.set_entry_point("model")

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")
graph.set_finish_point("end_node")



# Enable Interupt
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
print(display(Image(app.get_graph().draw_mermaid_png())))
thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}

linkedIn_topic = input("Enter your linkedIn topic: ")
initial_state:State ={
    "linkedIn_topic": linkedIn_topic,
    "generated_post": [],
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    print(chunk,"chunk")
    for node_id, value in chunk.items():
        if node_id == '__interupt__':
            while True:
                 user_feedback = input("Provide feedback (or type 'done' when finished): ")
                 app.invoke(Command(resume=user_feedback), config=thread_config)

                 if user_feedback.lower() == "done":
                    break


