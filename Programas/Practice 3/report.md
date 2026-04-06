# Technical Report
# Local RAG PDF Assistant

**Mario Tuset**

## 1. Problem Description

Accessing specific information from long documents such as PDFs or research papers can be difficult and is a time-consuming task. This application implements a local Question & Answering system based on RAG architecture that provides a graphical interface where users can upload their PDFs and ask questions to retrieve information from them. The system autonomously retrives the most relevant text fragments and sends them to a local LLM, generating answers grounden in the provided document.

## System desing and workflow

The system follows a multi-step pipeline that it is divided into two phases:

- Indexing Phase:
The user uploads a PDF via the Streamlit sidebar and the PyPDFLoader extracts the text from the document. Then, RecursiveCharacterTextSplitter divides the text into chunks, where each chunk is vectorized using nomic-embed-text via Ollama. All obtained vectors are stored in a local FAISS index.

- Retrieval and Generation Phase:
The user types a question in the Streamlit text input and it is vectorized using the same embedding model. FAISS performs a semantic similarity search and retrieves the top-k most relevant chunks and they are injected into a prompt template alongside the user's question. The local LLM generates the response and it is displayed in the Streamlit interface.
