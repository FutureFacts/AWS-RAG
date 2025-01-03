import os
import tempfile
import uuid
import json
import boto3
import faiss
import numpy as np
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import BedrockEmbeddings
from docx import Document
import warnings
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Suppress Streamlit warnings about missing ScriptRunContext
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# AWS Configuration (REPLACE WITH YOUR VALUES)
s3_client = boto3.client("s3", region_name="eu-central-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# FAISS Setup
faiss_index_file = "/tmp/faiss_index"
embedding_dimension = 1024  # Adjusted for model compatibility

# Bedrock Embeddings (REPLACE WITH YOUR VALUES)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())


def split_text(content, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(content)


def create_faiss_index(request_id, documents, bedrock_embeddings):
    try:
        st.info("Generating embeddings for documents...")
        embeddings_data = []

        for i, doc in enumerate(documents):
            try:
                embedding = bedrock_embeddings.embed_query(doc)
                embedding_np = np.array(embedding).astype(np.float32)
                if embedding_np.shape[0] != embedding_dimension:
                    st.warning(f"Skipping document {i + 1}: Embedding dimension mismatch (expected {embedding_dimension}, got {embedding_np.shape[0]})")
                    continue
                embeddings_data.append(embedding_np)
            except Exception as e:
                st.error(f"Error generating embedding for document {i + 1}: {e}")
                continue

        if not embeddings_data:
            st.error("No valid embeddings generated. Cannot create FAISS index.")
            return False

        embeddings_np = np.vstack(embeddings_data).astype(np.float32)

        st.info("Initializing FAISS index...")
        index = faiss.IndexFlatL2(embedding_dimension)
        index.add(embeddings_np)

        faiss.write_index(index, faiss_index_file)

        try:
            s3_key = f"{request_id}/faiss_index.index"
            with open(faiss_index_file, "rb") as f:
                s3_client.upload_fileobj(f, BUCKET_NAME, s3_key)
            st.success(f"FAISS index uploaded: s3://{BUCKET_NAME}/{s3_key}")
        except Exception as s3_e:
            st.error(f"Error uploading FAISS index to S3: {s3_e}")
            return False

        st.success("FAISS index created and uploaded to S3 successfully.")
        return True

    except Exception as e:
        import traceback
        st.error(f"Error creating FAISS index: {e}")
        traceback.print_exc()
        return False


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def main():
    st.title("Admin Site for Chat with Files")
    st.write("Streamlit app is running!")  # Debug message

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "json", "docx", "csv"], key="file_uploader_1")

    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request ID: {request_id}")
        saved_file_name = os.path.join(tempfile.gettempdir(), f"{request_id}_{uploaded_file.name}")

        with open(saved_file_name, "wb") as file:
            file.write(uploaded_file.getvalue())

        try:
            content = ""
            with st.spinner("Processing file..."):
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(saved_file_name)
                    pages = loader.load_and_split()
                    content = "\n".join(page.page_content for page in pages)
                elif uploaded_file.name.endswith(".txt"):
                    with open(saved_file_name, "r", encoding="utf-8") as file:
                        content = file.read()
                elif uploaded_file.name.endswith(".json"):
                    with open(saved_file_name, "r", encoding="utf-8") as file:
                        content = json.dumps(json.load(file), indent=2)
                elif uploaded_file.name.endswith(".docx"):
                    content = read_docx(saved_file_name)
                elif uploaded_file.name.endswith(".csv"):
                    loader = CSVLoader(saved_file_name)
                    content = "\n".join(str(row) for row in loader.load())

                st.success("File content loaded successfully.")

                splitted_docs = split_text(content, chunk_size=1000, chunk_overlap=200)
                st.write(f"Splitted Docs Length: {len(splitted_docs)}")
                if splitted_docs:
                    st.write(splitted_docs[:2])

                    st.info("Creating the FAISS Index...")
                    if create_faiss_index(request_id, splitted_docs, bedrock_embeddings):
                        st.success("File processed and FAISS index created successfully.")
                    else:
                        st.error("Failed to process file.")
                else:
                    st.warning("No content to process after splitting.")

        finally:
            os.remove(saved_file_name)


if __name__ == "__main__":
    main()
