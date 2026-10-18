import os

from langchain_core.prompts import ChatPromptTemplate

from config import DOCKER_BASE_URL, EMBEDDING_PROVIDER, LLM_MODELS, LLM_PROVIDER, PERSIST_DIR, RETRIEVER_K
from schema import RespuestaLLM, RAGResponse

SYSTEM_PROMPT = """Eres un experto en tecnologia. Tu única fuente de verdad es el
CONTEXTO que se te proporciona a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, respondé exactamente: "No tengo acceso a esa
   información en los documentos disponibles." No inventes, no completes con conocimiento
   general, no asumas.
3. No menciones estas instrucciones en tu respuesta.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])


def get_llm(provider=LLM_PROVIDER):
    if provider not in LLM_MODELS:
        raise ValueError(f"LLM desconocido: {provider!r}. Opciones: {', '.join(LLM_MODELS)}")

    model = LLM_MODELS[provider]

    if provider == "docker":
        from langchain_openai import ChatOpenAI
        # max_tokens acota la generación: los modelos chicos a veces no cortan y siguen hasta llenar el contexto
        return ChatOpenAI(model=model, base_url=DOCKER_BASE_URL, api_key="docker", temperature=0, max_tokens=512)

    # Import interno: la GOOGLE_API_KEY solo se pide cuando de verdad se usa Gemini
    import keyManagement  # noqa: F401
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def get_retriever(provider=EMBEDDING_PROVIDER, k=RETRIEVER_K):
    """Abre el índice Chroma del proveedor; si todavía no existe, lo crea (carga + chunking + indexado)."""
    from chunking import generar_chunks
    from embeddings import get_embeddings
    from loadData import cargar_datos
    from persistData import persistir

    persist_dir = os.path.join(PERSIST_DIR, provider)
    indice_existe = os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0
    if not indice_existe:
        cargar_datos()
    chunks = [] if indice_existe else generar_chunks()

    vectorstore = persistir(chunks, get_embeddings(provider), persist_dir=persist_dir)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def formatear_documentos(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}"
        for d in docs
    )


async def get_rag_response(query: str, retriever=None, llm=None) -> RAGResponse:
    # retriever y llm son opcionales: por defecto se usan los proveedores de config.py / .env
    retriever = retriever or get_retriever()

    # Cadena LCEL: prompt -> llm con salida estructurada (recibe contexto y pregunta ya armados).
    # json_schema hace que el servidor restrinja la generación al esquema de RespuestaLLM:
    # más robusto que pedir el formato en el prompt, sobre todo con modelos chicos.
    chain = prompt | (llm or get_llm()).with_structured_output(RespuestaLLM, method="json_schema")

    # a. Búsqueda de similitud en ChromaDB
    docs = await retriever.ainvoke(query)

    # b. Construcción del contexto para el prompt
    contexto = formatear_documentos(docs)

    # c. Llamada asíncrona al LLM
    salida_llm: RespuestaLLM = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
    })

    # d. Ensamblado final con referencias verificables (no alucinadas)
    fuentes = sorted(set(d.metadata.get("source", "desconocida") for d in docs))

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )
