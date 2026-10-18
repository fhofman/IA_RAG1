from config import DOCKER_BASE_URL, EMBEDDING_MODELS


def get_embeddings(provider):
    """Devuelve el modelo de embeddings del proveedor elegido.

    Los imports son internos para que solo se necesite instalado
    (y configurado) el proveedor que se usa.
    """
    if provider not in EMBEDDING_MODELS:
        raise ValueError(f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(EMBEDDING_MODELS)}")

    model = EMBEDDING_MODELS[provider]

    if provider == "gemini":
        import keyManagement  # noqa: F401  pide la GOOGLE_API_KEY si no está definida
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "docker":
        from langchain_openai import OpenAIEmbeddings
        # Docker Model Runner no valida la api_key; check_embedding_ctx_length=False
        # manda texto plano en vez de tokens de tiktoken (que solo entiende OpenAI)
        return OpenAIEmbeddings(model=model, base_url=DOCKER_BASE_URL, api_key="docker",
                                check_embedding_ctx_length=False)

    from langchain_huggingface import HuggingFaceEmbeddings
    # CPU: el modelo es chico y así no compite por la memoria de GPU con el LLM local
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})


def generar_embeddings(provider):
    """Prepara el modelo de embeddings; los vectores los calcula Chroma al indexar."""
    embeddings = get_embeddings(provider)
    print(f"🧠 Proveedor: {provider} — modelo: {EMBEDDING_MODELS[provider]}")

    dimension = len(embeddings.embed_query("prueba"))
    print(f"🔢 Dimensión de los vectores: {dimension}")
    return embeddings
