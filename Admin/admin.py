import os
import tempfile
import uuid
import json
import boto3
import streamlit as st
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import BedrockEmbeddings
from docx import Document

# AWS Configuration (REPLACE WITH YOUR VALUES)
s3_client = boto3.client("s3", region_name="eu-central-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# LanceDB Setup
lancedb_base_path = "/tmp/lancedb"  # Or your preferred path
os.makedirs(lancedb_base_path, exist_ok=True)
lancedb_path = os.path.join(lancedb_base_path, "lancedb")
os.makedirs(lancedb_path, exist_ok=True)

# Bedrock Embeddings (REPLACE WITH YOUR VALUES)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())


def split_text(content, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(content)


def create_vector_store(request_id, documents, bedrock_embeddings):
    try:
        # Generate embeddings for each document
        embeddings_data = []
        for doc in documents:
            embedding = bedrock_embeddings.embed_query(doc)
            embeddings_data.append({"id": get_unique_id(), "page_content": doc, "embedding": embedding})

        # Store all data in a single table
        table_name = "vector_store"
        db = lancedb.connect(lancedb_path)

        # Check if the table already exists
        if table_name in db.table_names():
            table = db.open_table(table_name)
            table.add(embeddings_data)  # Append new data to the existing table
        else:
            db.create_table(table_name, embeddings_data)  # Create a new table

        # Verify the table exists
        table_file_path = os.path.join(lancedb_path, f"{table_name}.lance")
        if not os.path.exists(table_file_path):
            st.error(f"Vector store table file does not exist: {table_file_path}")
            return False

        # Upload LanceDB files to S3 without compression
        for root, dirs, files in os.walk(lancedb_path):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f"{request_id}/{os.path.relpath(file_path, lancedb_path)}"
                try:
                    with open(file_path, 'rb') as f:
                        s3_client.upload_fileobj(f, BUCKET_NAME, s3_key)
                    st.write(f"Uploaded: s3://{BUCKET_NAME}/{s3_key}")
                except Exception as s3_e:
                    st.error(f"Error uploading file {file} to S3: {s3_e}")
                    return False

        st.success("Data stored in LanceDB and uploaded to S3 successfully.")
        return True

    except Exception as e:
        import traceback
        st.error(f"Error creating LanceDB vector store: {e}")
        traceback.print_exc()
        return False


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def main():
    st.title("Admin Site for Chat with Files")

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

                    st.info("Creating the Vector Store...")
                    if create_vector_store(request_id, splitted_docs, bedrock_embeddings):
                        st.success("File processed and vector store created successfully.")
                    else:
                        st.error("Failed to process file.")
                else:
                    st.warning("No content to process after splitting.")

        finally:
            os.remove(saved_file_name)


if __name__ == "__main__":
    main()
