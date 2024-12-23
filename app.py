import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

# Page configuration
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("Chat with Your PDF")

# Initialize components (using cache to avoid reloading)
@st.cache_resource
def initialize_rag():
    # Initialize Ollama
    llm = Ollama(model="llama2")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load PDF
    loader = PyPDFLoader("./data/1.5.1. Computing Related Legislation.pdf")
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize RAG system
qa_chain = initialize_rag()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG
    with st.chat_message("assistant"):
        response = qa_chain({"question": prompt, "chat_history": []})
        st.markdown(response["answer"])
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})