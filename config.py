import os

from dotenv import load_dotenv

load_dotenv()

# Proveedor de embeddings a usar: "gemini", "docker" o "huggingface"
# Se puede sobrescribir en el .env (EMBEDDING_PROVIDER) o al ejecutar: python main.py docker
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

# ⚠️ Mismo modelo de embeddings para indexar Y para consultar
EMBEDDING_MODELS = {
    "gemini": "models/gemini-embedding-001",
    "docker": "ai/embeddinggemma",
    "huggingface": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Docker Model Runner: API compatible con OpenAI (docker desktop enable model-runner --tcp 12434)
DOCKER_BASE_URL = "http://localhost:12434/engines/v1"

# Base vectorial (Chroma). Se guarda una subcarpeta por proveedor, porque los
# vectores de modelos distintos no son compatibles: vectorstore/<proveedor>/
PERSIST_DIR = "vectorstore"
COLLECTION_NAME = "productos"

# LLM que genera la respuesta: "gemini" o "docker"
# Se puede sobrescribir en el .env (LLM_PROVIDER) o al ejecutar: python main.py huggingface "" docker
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODELS = {
    "gemini": "gemini-flash-latest",
    "docker": "ai/llama3.2:3B-Q4_K_M",  # alternativa más liviana: "ai/llama3.2:1B-Q8_0"
}

# Recuperación y generación de la respuesta
RETRIEVER_K = 4
