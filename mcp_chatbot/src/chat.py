import os
from langchain_ollama import ChatOllama

def get_chat_model():
    """
    Returns a ChatOllama instance configured from environment variables.
    """
    base_url = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL")

    if not base_url or not model:
        raise ValueError("OLLAMA_BASE_URL and OLLAMA_MODEL environment variables must be set.")

    return ChatOllama(base_url=base_url, model=model) 