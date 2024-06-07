# # Main Code
# from langchain_community.utilities import SerpAPIWrapper
# from langchain.agents import load_tools
# from dotenv import load_dotenv
# import os
# from key import cohere_api_key

# # Load environment variables
# load_dotenv()

# from langchain.agents import AgentExecutor
# from langchain_cohere.chat_models import ChatCohere
# from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain.tools.retriever import create_retriever_tool

# # RAGS
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_cohere import CohereEmbeddings
# from langchain_community.vectorstores import FAISS
# from io import BytesIO 

# # Create and run the Cohere agent
# llm = ChatCohere()

# # First RAG Agent:CV
# def cv_rag_agent(uploaded_file):
#     # Load and split text documents
#     if uploaded_file.name.endswith(".docx"):
#         file_data = uploaded_file.read()
#         loader = Docx2txtLoader(BytesIO(file_data))
#         data = loader.load()
#     if uploaded_file.name.endswith(".pdf"):
#         loader = PyPDFLoader(BytesIO(file_data))
#         data = loader.load()
#     return data

#     # splitter = RecursiveCharacterTextSplitter(
#     #     chunk_size=500,
#     #     chunk_overlap=50
#     # )
#     # splits = splitter.split_documents(pages)
#     # embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
#     # vector_store = FAISS.from_documents(splits, embeddings)
#     # retrieval = vector_store.as_retriever(search_kwargs={'k': 3})

#     # retreiver_tool = create_retriever_tool(
#     #     retrieval,
#     #     "search_the_scripture",
#     #     "Use this tool when the context of a user's question is on chrsitianity."
#     # )






# # Streamlit Interface
# import streamlit as st
# st.set_page_config(page_title="Cover Letter Writter", page_icon='🧾')

# with st.sidebar:
#     st.header("References")
#     st.text_input("Paste Job Post Url")
#     uploaded_file = st.file_uploader("Upload CV File", type=['pdf', 'docx'])

#     st.button('Generate Cover letter')

# # Create a container for the card box
# card_box = st.container()

# with card_box:
#     # Add a title for the card
#     st.header("Cover Letter")
#     if uploaded_file:
#         content = cv_rag_agent(uploaded_file)
#     # Add content to the card body
#         st.write(content)

#     # Add a button or link (optional)
#     if st.button("Export"):
#         # Perform an action when the button is clicked
#         pass


# main.py

# main.py
# main.py
# main.py

from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from dotenv import load_dotenv
import os
from key import cohere_api_key

# Load environment variables
load_dotenv()

from langchain.agents import AgentExecutor
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool

# RAGS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from io import BytesIO

# Create and run the Cohere agent
llm = ChatCohere()

# First RAG Agent: CV
def cv_rag_tool(uploaded_file):
    # Read file data
    file_data = uploaded_file.read()
    file_name = uploaded_file.name
    
    # Load and extract text from documents
    if file_name.endswith(".docx"):
        with open('temp.docx', 'wb') as temp_file:
            temp_file.write(file_data)
        loader = Docx2txtLoader('temp.docx')
        data = loader.load()
        os.remove('temp.docx')
        text = '\n'.join([page.page_content for page in data])
    elif file_name.endswith(".pdf"):
        with open('temp.pdf', 'wb') as temp_file:
            temp_file.write(file_data)
        loader = PyPDFLoader('temp.pdf')
        data = loader.load()
        os.remove('temp.pdf')
        text = '\n'.join([page.page_content for page in data])
    else:
        raise ValueError("Unsupported file type")
    
    # Create Document objects
    document = Document(page_content=text)
    
    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents([document])
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    vector_store = FAISS.from_documents(splits, embeddings)
    retrieval = vector_store.as_retriever(search_kwargs={'k': 3})

    retreiver_tool = create_retriever_tool(
        retrieval,
        "cv_tool",
        "Use this tool to see my CV."
    )
    return retreiver_tool
    

# Streamlit Interface
import streamlit as st
st.set_page_config(page_title="Cover Letter Writer", page_icon='🧾')

with st.sidebar:
    st.header("References")
    job_url = st.text_input("Paste Job Post Url")
    uploaded_file = st.file_uploader("Upload CV File", type=['pdf', 'docx'])

    if st.button('Generate Cover letter') and uploaded_file:
        try:
            content = cv_rag_tool(uploaded_file)
            st.write(content)
        except ValueError as e:
            st.error(str(e))

# Create a container for the card box
card_box = st.container()

with card_box:
    # Add a title for the card
    st.header("Cover Letter")
    if uploaded_file:
        try:
            content = cv_rag_tool(uploaded_file)
            # Add content to the card body
            st.write(content)
        except ValueError as e:
            st.error(str(e))

    # Add a button or link (optional)
    if st.button("Export"):
        # Perform an action when the button is clicked
        pass
