import os
import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# -----------------------------
# Load Environment Variables
# -----------------------------
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# -----------------------------
# Initialize LLM
# -----------------------------
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key= os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Initialize ChromaDB
# -----------------------------
client = chromadb.Client()

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="career_knowledge_base",
    embedding_function=embedding_function
)

# -----------------------------
# Function to ingest PDF
# -----------------------------
def ingest_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Career Guidance Chatbot")
st.markdown("Get personalized career advice using RAG + LLM.")

# Prevent multiple ingestion
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

# -----------------------------
# Upload PDFs
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload career resource files (PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Ingest PDFs
# -----------------------------
if uploaded_files and not st.session_state.db_loaded:

    for file in uploaded_files:

        text = ingest_pdf(file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = splitter.split_text(text)

        batch_size = 50

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i+batch_size]

            ids = [
                f"{file.name}_{j}"
                for j in range(i, i + len(batch))
            ]

            collection.add(
                documents=batch,
                ids=ids
            )

    st.session_state.db_loaded = True
    st.success(f"{len(uploaded_files)} PDFs ingested successfully!")

# -----------------------------
# User Query
# -----------------------------
user_query = st.text_input("Ask your career question:")

if st.button("Get Advice") and user_query:

    with st.spinner("Analyzing resources and generating advice..."):

        # -----------------------------
        # Vector Search
        # -----------------------------
        vector_results = collection.query(
            query_texts=[user_query],
            n_results=5
        )

        vector_docs = vector_results["documents"][0]

        # -----------------------------
        # Keyword Search
        # -----------------------------
        keywords = user_query.lower().split()

        keyword_docs = [
            doc for doc in vector_docs
            if any(k in doc.lower() for k in keywords)
        ]

        # -----------------------------
        # Hybrid Merge
        # -----------------------------
        hybrid_docs = list(set(vector_docs + keyword_docs))

        # -----------------------------
        # Reranking with LLM
        # -----------------------------
        rerank_prompt = PromptTemplate.from_template(
        """
        User Query:
        {query}

        Documents:
        {docs}

        Rank these documents from most relevant to least relevant
        for providing career advice including skills and companies.

        Return the ranked list.
        """
        )

        rerank_chain = rerank_prompt | llm

        reranked_output = rerank_chain.invoke({
            "query": user_query,
            "docs": hybrid_docs
        })

        top_context = reranked_output.content.split("\n")[:3]

        # -----------------------------
        # Final RAG Prompt
        # -----------------------------
        final_prompt = PromptTemplate.from_template(
        """
        You are a Career Guidance AI assistant.

        Based on the following resources:

        {context}

        Provide a personalized roadmap for the user:

        - Skills to learn
        - Recommended companies
        - Steps to improve career readiness

        User Query:
        {query}

        Provide a clear and structured answer.
        """
        )

        rag_chain = final_prompt | llm

        career_advice = rag_chain.invoke({
            "context": "\n".join(top_context),
            "query": user_query
        })

        # -----------------------------
        # Display Results
        # -----------------------------
        st.subheader("Top Retrieved Context")

        for doc in top_context:
            st.write("-", doc)

        st.subheader("Personalized Career Advice")

        st.write(career_advice.content)