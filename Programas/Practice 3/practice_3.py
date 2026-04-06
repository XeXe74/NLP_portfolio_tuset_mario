import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# UI Configuration
st.set_page_config(page_title="RAG PDF Assistant", layout="centered")
st.title("📚 Local RAG PDF Assistant")
st.write("Upload a PDF document to query its content using local LLMs.")

# Cached to avoid reloading models on every Streamlit interaction
@st.cache_resource
def load_models():
    """
    Load the embedding and LLM models from Ollama.
    """
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    llm_model = Ollama(model="llama3") # Llama3 model
    return embed_model, llm_model

embeddings, llm = load_models() # Load models once and reuse


# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF file", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Process PDF and split into chunks
    with st.spinner("Extracting and splitting text from PDF..."):
        # Load PDF content
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split into chunks for LLM context window
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create vector store for retrieval
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Set up RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        st.success(f"PDF processed into {len(splits)} text chunks.")

    # Clean up temporary file
    os.remove(tmp_file_path)

    # Input for user query
    query = st.text_input("Ask a question about the document:")

    # Generate response from the RAG chain
    if query:
        with st.spinner("Generating response..."):
            response = qa_chain.invoke(query)
            st.markdown("### 🤖 Assistant Response:")
            st.info(response["result"])