"""Embeddings functionality"""
from sentence_transformers import SentenceTransformer


embeddings_model = SentenceTransformer("all-mpnet-base-v2")

def get_dimensions() -> int:
    """
    Get the number of dimensions for the embeddings.

    Returns:
        int: The number of dimensions.

    """
    return embeddings_model.get_sentence_embedding_dimension()

def get_embeddings(string: str) -> list[float]:
    """
    Get the embeddings for the given query.

    Args:
        string (str): The string to get embeddings for.

    Returns:
        list: The embeddings for the string.

    """
    return embeddings_model.encode(string).tolist()
