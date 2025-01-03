import boto3
from config import BUCKET_NAME, folder_path
from langchain_community.vectorstores import FAISS
from get_embedding import get_embedding_function

# s3_client
s3_client = boto3.client("s3")


def load_index():
    """
    Downloads the FAISS index and associated files from the
    specified S3 bucket to a local directory. This function
    retrieves two files:
    - my_faiss.faiss: The main FAISS index file.
    - my_faiss.pkl: The associated metadata file for the FAISS index.

    Downloads the files to the local directory specified by `folder_path`.

    Example Usage:
        load_index()
    """
    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss",
        Filename=f"{folder_path}my_faiss.faiss",
    )
    s3_client.download_file(
        Bucket=BUCKET_NAME, Key="my_faiss.pkl",
        Filename=f"{folder_path}my_faiss.pkl"
    )


def create_index():
    """
    Creates and loads a FAISS index from the local directory using embeddings
    generated by the AWS Bedrock model through the `get_embedding_function()`.

    The function loads the FAISS index using the `FAISS.load_local()` method,
    specifying the index name, local folder path, and embedding function.

    Returns:
        FAISS: The loaded FAISS index that can be used for similarity searches.

    Example Usage:
        faiss_index = create_index()
    """
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        # Retrieves the embedding function for Bedrock
        embeddings=get_embedding_function(),
        # Allow deserialization of potentially unsafe data
        allow_dangerous_deserialization=True,
    )
    return faiss_index
