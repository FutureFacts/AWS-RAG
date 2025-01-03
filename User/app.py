import streamlit as st
import uuid
from langchain.llms.bedrock import Bedrock

# Custom functions for FAISS indexing
from load_faiss_index import load_index, create_index

# Config file with necessary credentials
from config import llm_model_id, BUCKET_NAME, aws_bedrock_client

from get_answer import get_response1


# Function to generate a unique ID
def get_unique_id():
    return str(uuid.uuid4())


# Function to get the LLM (Language Model) from AWS Bedrock
def get_llm():
    return Bedrock(model_id=llm_model_id, client=aws_bedrock_client)


# Main function to handle the Streamlit app
def main():
    st.header(f"Chat with PDF demo ({BUCKET_NAME})")
    # Load and create the FAISS index for querying
    load_index()
    faiss_index = create_index()

    st.write("INDEX IS READY")

    # User inputs a question
    question = st.text_input("Ask your question")

    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            # Get the LLM and generate a response
            llm = get_llm()
            response = get_response1(llm, faiss_index, question)
            # Display the response
            st.write(response)
            st.success("Done")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
