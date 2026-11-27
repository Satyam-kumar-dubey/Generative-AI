
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Annotated

llm = ChatGroq(model="openai/gpt-oss-120b")
memory = InMemorySaver()

class Chatstate(BaseModel):
    messages : Annotated[list, add_messages]
    
def chatbotnode(state:Chatstate) -> Chatstate:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

graph = StateGraph(Chatstate)
graph.add_node("chatbot", chatbotnode)

graph.add_edge(START,"chatbot")
graph.add_edge("chatbot", END)

finalgraph = graph.compile(checkpointer= memory)

while True:
    
    query = input("User: ")
    if query.lower() in ["quit","bye","exit"]:
        print("AI: ", "Bye...")
        break
    
    if query:
        res = finalgraph.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":"abc123"}}
        )
        ans = res["messages"][-1].content
        print("AI: ",ans)