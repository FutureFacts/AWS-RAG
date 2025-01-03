"""
Import the Boto3 library,
which is used for interacting with AWS services
"""

import boto3

# Define a temporary folder path where Faiss files stored or processed
folder_path = "/tmp/"

# Define the AWS region where the resources will be deployed and accessed
region_name = "eu-central-1"

# The name of the S3 bucket where Faiss files stored or accessed
BUCKET_NAME = "ip24-rag-gen"

# Define the model ID for the embedding model, used for embedding text
embeddings_model_id = "amazon.titan-embed-text-v2:0"
"""
Define the model ID for the LLM (Large Language Model),
that processes and generates text
"""
llm_model_id = "amazon.titan-text-lite-v1"
"""
Initialize the AWS Bedrock client.
This client will allow interaction with the Bedrock service.
Bedrock is used to run large language models (LLMs)
and other AI services from AWS.
"""
aws_bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
)
