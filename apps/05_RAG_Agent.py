
from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool 
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_doc = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = InMemoryVectorStore.from_documents(
        embedding=embeddings,
        documents=splitted_doc
    )


    @tool
    def retriver_tool(query : str):
        """
            This tool is used to retrieve relevant documents from the vector database based on the user's query.
        """
        
        data = vector_store.similarity_search(query=query)
        context = ""
        for chunk in data:
            context += chunk.page_content + "\n"
            
        return context


    llm = ChatGroq(model="openai/gpt-oss-20b")

    system_prompt = """You are a helpful assistant that answers questions using retriver tool. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retriver_tool` tool for questions requiring external knowledge."""
        
    agent = create_agent(
        
        model=llm,
        tools=[retriver_tool],
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt,
    )
    
    st.session_state.document_uploaded = True
    st.session_state.agent = agent
    
    
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload Your Pdf Here", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path ="./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            
            process_document(path)
            st.rerun()
    

if st.session_state.document_uploaded and st.session_state.agent:
    
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)
    
    query = st.chat_input("Ask me anything...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role":"user","content":query})
        
        res = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":"1"}}
        )
        
        ans = res["messages"][-1].content
        st.chat_message("ai").markdown(ans)
        st.session_state.messages.append({"role":"ai","content":ans})
    
    
    
    

