# Career Guidance Chatbot

This project implements a **Career Guidance Chatbot** that helps users get personalized career advice based on uploaded PDF resources. The system uses **Retrieval Augmented Generation (RAG)** to retrieve relevant information from documents and generate intelligent responses using a Large Language Model (LLM).

Users can upload career-related PDF files, ask questions, and receive suggestions such as skills to learn, recommended companies, and steps to improve career readiness.

## Features
- Upload and analyze career-related PDF documents
- Semantic search using vector embeddings
- Hybrid retrieval with vector search and keyword search
- AI-generated career guidance and recommendations
- Interactive web interface

## Technologies Used
- Python
- Streamlit
- LangChain
- ChromaDB
- Groq LLM
- Sentence Transformers

## How It Works
1. Users upload career-related PDF documents.
2. The system extracts text and splits it into smaller chunks.
3. Text chunks are converted into embeddings and stored in a vector database.
4. When a user asks a question, the system retrieves the most relevant documents.
5. The LLM analyzes the retrieved context and generates personalized career advice.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
