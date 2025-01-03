from config import embeddings_model_id, aws_bedrock_client
from langchain_community.embeddings import BedrockEmbeddings


def get_embedding_function():
    """
    Initializes and returns an embedding function using AWS Bedrock.

    This function creates an instance of the `BedrockEmbeddings` class from
    Langchain Community,which interfaces with an embedding model hosted
    on AWS Bedrock.The function requires two pieces of information: the
    model ID (`embeddings_model_id`) and a configured
    AWS Bedrock client (`aws_bedrock_client`).

    Returns:
        BedrockEmbeddings: An instance of the `BedrockEmbeddings` class,
        which can be used to generate embeddings for input text
        using the specified AWS Bedrock model.

    Example Usage:
        embedding_function = get_embedding_function()
        embeddings = embedding_function.embed_text("Hello, world!")
    """
    # Retrieve the AWS Bedrock client and model ID
    bedrock_client = aws_bedrock_client
    bedrock_embeddings = BedrockEmbeddings(
        model_id=embeddings_model_id, client=bedrock_client
    )

    # Return the embedding function
    return bedrock_embeddings
