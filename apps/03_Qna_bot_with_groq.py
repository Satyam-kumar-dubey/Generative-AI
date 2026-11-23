
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st;

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an ai agent and can search the queries on the google as well"
)

st.subheader("QNA chatBot : Get Answer as faster as Your Thought")
query = st.chat_input("Enter Your Query..")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    
    res = agent.stream(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode= "messages"
    )
    
    ai_container = st.chat_message("ai")
    with ai_container:
        
        space = st.empty()
        message = ""
        
        for chunk in res:
            message = message + chunk[0].content
            space.write(message)
    
        st.session_state.history.append({"role":"ai","content":message})




