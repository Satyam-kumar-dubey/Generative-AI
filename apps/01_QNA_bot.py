
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("🤖 AskBuddy: AI QNA Bot")
st.markdown("My QnA Bot with Google Gemini and LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything..")

if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    if isinstance(res.content, list):
                answer = "\n".join(
                    block["text"]
                    for block in res.content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                )
    else:
        answer = res.content

    st.chat_message("ai").markdown(answer)
    st.session_state.messages.append({"role": "ai","content": answer})

