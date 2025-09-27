import os
import streamlit as st

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader # Corrected
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Corrected
from langchain_community.vectorstores import FAISS # Corrected
from langchain_community.llms import HuggingFacePipeline # Corrected
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- UI Configuration ---
st.set_page_config(page_title="AI Study Buddy 🧠", layout="wide")
st.title("Personalised AI Study Buddy 🧠")
st.write("Upload your lecture notes and textbook PDFs, then ask complex questions!")

# --- Global Variables & Constants ---
DATA_DIR = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Helper Functions ---

# Use Streamlit's caching to load the model and create the pipeline only once
@st.cache_resource
def setup_qa_chain():
    """
    Sets up the entire RAG pipeline:
    1. Loads documents from the 'data' directory.
    2. Chunks the documents into smaller pieces.
    3. Creates embeddings using Sentence-Transformers.
    4. Stores embeddings in a FAISS vector database.
    5. Loads a pre-trained language model from Hugging Face.
    6. Creates a RetrievalQA chain.
    """
    # 1. Load documents
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            docs.extend(loader.load())

    if not docs:
        st.warning("No PDF documents found in the 'data' directory. Please upload some files.", icon="⚠️")
        return None

    # 2. Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings
    # We use a powerful but efficient model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # 4. Store embeddings in FAISS vector database
    # This creates a searchable index of our document chunks
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

    # 5. Load a pre-trained Large Language Model (LLM) from Hugging Face
    # We use Flan-T5, a great model for question answering
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    
    # Create the pipeline
    llm = HuggingFacePipeline.from_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    # 6. Create the RetrievalQA chain
    # This chain links the retriever (our vector store) and the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff', # 'stuff' means all retrieved text is "stuffed" into the prompt
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain


# --- Main Application Logic ---

# Check if the 'data' directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF files here", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the file to the 'data' directory
            with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

# Initialize the QA chain
qa_chain = setup_qa_chain()

# Main interface for asking questions
st.header("Ask a Question")
user_question = st.text_input("What would you like to know from your documents?")

if user_question:
    if qa_chain:
        with st.spinner("Searching for answers..."):
            try:
                # Run the query through the RAG chain
                response = qa_chain.run(user_question)
                st.write("### Answer")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload at least one PDF to begin.", icon="💡")