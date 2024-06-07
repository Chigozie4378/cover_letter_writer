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
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from io import BytesIO

# Create and run the Cohere agent
llm = ChatCohere()

# First RAG Agent: CV
def cv_rag_retriever(uploaded_file):
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
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})

    return retriever
    
# First RAG Agent: CV
def job_description_rag_retriever(url):    
    web_loader = UnstructuredURLLoader(urls=[url])
    document = web_loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = splitter.split_documents(document)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})

    
    return retriever




#==============================================================================

def process(cv_retreiver_tool,job_description_retreiver_tool,cv_retriever,job_description_retriever):
    # Updated prompt to include a context placeholder for the retrieved text
    prompt = ChatPromptTemplate.from_template("""
    You are professional cover letter writer. 
    You must use my CV and Job Description to generate a cover letter: 
    CV: {context1}
    Job Description: {context2}
    Question: {question}
    Follow the template below when writing the cover letter based on my CV and Job's Description .
    [Date]
    First Name Surname
    Hiring Manager's
    Company name)
    Street address]
    (City/Town]. [Postcode]
    [Phone]
    [Emai]
    RE:[Job Title]. [Reference Code]
    Dear Hiring manager,
    I am writing to express my interest in the vacant [Job Tale] position at [Company Name] which was advertised on [Job Advertisement Source]
    As a skilled [Your Profession] with [# years] of experience in [relevant job industry]. I believe my background and skills make me well-suited for this role
    In my previous position as [Recent Job Title] at [Previous Company Name]. I had the opportunity to [describe a responsibility, task, or project you completed and what you achieved) This experience has [explain how this experience is relevant to the job post]
    What excites me most about the prospect of working at [Company flame] is [mention something you admire about the company or how you relate to their values or mission] I believe my [specific skills] align perfectly with your company's goals, and I am confident that I could bring value to your team
    Thank you for considering my application. I look forward to the possibility of discussing this exciting opportunity with you. I am readily available for an interview at your earliest
    convenience
    Yours sincerely
    [Your Name]
  
    
    """)

    # Function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    agent = create_cohere_react_agent(
        llm=llm,
        tools=[cv_retreiver_tool,job_description_retreiver_tool],
        prompt=prompt,
    )

    # chat_history = [
    #     HumanMessage(content="My name is Sunday?"),
    #     AIMessage(content="Hi Sunday, Nice to meet you!"),
    # ]

    agent_executor = AgentExecutor(agent=agent, tools=[cv_retreiver_tool,job_description_retreiver_tool], verbose=True)


    user_input = "Generate a cover letter using provided CV and job description"

    # Retrieve documents related to the query

    retrieved_cv = cv_retriever.get_relevant_documents(user_input)
    formatted_cv = format_docs(retrieved_cv)

    retrieved_job_description = job_description_retriever.get_relevant_documents(user_input)
    formatted_job_description = format_docs(retrieved_job_description)

    response = agent_executor.invoke({
        "question": user_input,
        "context1": formatted_cv,  # Add the retrieved context here
        "context2": formatted_job_description  # Add the retrieved context here
    })
        
        
    # chat_history.append(HumanMessage(content=user_input))
    # chat_history.append(AIMessage(content=response['output']))
    
    # See Cohere's response
    return response.get("output")

#==============================================================================


# Streamlit Interface
import streamlit as st
st.set_page_config(page_title="Cover Letter Writer", page_icon='🧾')

with st.sidebar:
    st.header("References")
    job_url = st.text_input("Paste Job Post Url")
    uploaded_file = st.file_uploader("Upload CV File", type=['pdf', 'docx'])

    if st.button('Generate Cover letter'):
        try:
            st.write('chain here')
        except ValueError as e:
            st.error(str(e))

# Create a container for the card box
card_box = st.container()

with card_box:
    # Add a title for the card
    st.header("Cover Letter")
    if 'result' in st.session_state:
        st.write('hi')
        st.write(st.session_state.result)

    # Add a button or link (optional)
    if st.button("Export"):
        # Perform an action when the button is clicked
        pass
