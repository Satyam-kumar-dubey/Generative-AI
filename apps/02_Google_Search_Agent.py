
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = InMemorySaver()

from langchain.agents import create_agent
agent = create_agent(
    model= llm,
    tools=[search.run],
    checkpointer= memory,
    system_prompt= "You are an agent that can search the queries on the google"
)

while True:
    
    query = input("User: ")
    if query.lower() == "exit":
        print("AI: ", "GoodBye..")
        break
    
    res = agent.invoke({"messages":[{"role":"user", "content":query}]},{"configurable": {"thread_id":"1"}},)
    print("AI: ", res['messages'][-1].content)  