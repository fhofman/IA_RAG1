import asyncio
import os
import sys

from config import EMBEDDING_MODELS, EMBEDDING_PROVIDER, LLM_MODELS, LLM_PROVIDER, PERSIST_DIR, RETRIEVER_K
from loadData import cargar_datos
from chunking import generar_chunks
from embeddings import generar_embeddings
from persistData import persistir
from response import get_llm, get_rag_response

# Preguntas de prueba: las primeras tienen la respuesta en los documentos;
# las últimas no, y el LLM debería decir que no tiene acceso a esa información.
PREGUNTAS = [
    # Auralis X2 (auriculares)
    "¿Cuánto dura la batería de los Auralis X2 con ANC activado?",
    "¿Cuánto cuesta el plan Sonora Care+ y qué agrega?",
    # Krono R9 (robot aspiradora)
    "¿Qué significa el Error 6 del Krono R9?",
    "¿Cuál es el teléfono de soporte de Hogaria?",
    # Vértice Book 14 Ultra (notebook)
    "¿Se puede ampliar la RAM de la Vértice Book 14 Ultra?",
    "¿Cuántos monitores externos admite la Vértice Book 14 Ultra?",
    # Fuera del contexto: tema general
    "¿Cuál es la capital de Francia?",
    # Fuera del contexto: producto que no está en los documentos
    "¿Cuánto dura la batería del iPhone 16 Pro?",
    "¿Cuál es la potencia de succión del Krono R11?",
]


async def responder(preguntas, retriever, llm):
    respuestas = []
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n❓ [{i}/{len(preguntas)}] {pregunta}")
        respuesta = await get_rag_response(pregunta, retriever, llm=llm)
        print(respuesta.model_dump_json(indent=2))
        respuestas.append(respuesta)
    return respuestas


def main(provider, preguntas, llm_provider):
    if provider not in EMBEDDING_MODELS:
        sys.exit(f"❌ Proveedor desconocido: {provider!r}. Opciones: {', '.join(EMBEDDING_MODELS)}")
    if llm_provider not in LLM_MODELS:
        sys.exit(f"❌ LLM desconocido: {llm_provider!r}. Opciones: {', '.join(LLM_MODELS)}")

    print("=== 1. Carga de datos ===")
    cargar_datos()

    print("\n=== 2. Chunking ===")
    chunks = generar_chunks()

    print("\n=== 3. Embeddings ===")
    embeddings = generar_embeddings(provider)

    print("\n=== 4. Persistencia (Chroma) ===")
    vectorstore = persistir(chunks, embeddings, persist_dir=os.path.join(PERSIST_DIR, provider))

    print("\n=== 5. Respuestas (RAG) ===")
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    print(f"🤖 LLM: {llm_provider} — modelo: {LLM_MODELS[llm_provider]}")
    # Un solo asyncio.run para todas las preguntas (el cliente HTTP del LLM queda atado a ese loop)
    respuestas = asyncio.run(responder(preguntas, retriever, get_llm(llm_provider)))

    print("\n🏁 Pipeline completo")
    return respuestas


if __name__ == "__main__":
    # python main.py [gemini|docker|huggingface] ["pregunta"] [gemini|docker] — sin argumentos usa config.py
    # Sin pregunta (o con "") se usan todas las PREGUNTAS de prueba.
    provider = sys.argv[1] if len(sys.argv) > 1 else EMBEDDING_PROVIDER
    preguntas = [sys.argv[2]] if len(sys.argv) > 2 and sys.argv[2] else PREGUNTAS
    llm_provider = sys.argv[3] if len(sys.argv) > 3 else LLM_PROVIDER
    main(provider, preguntas, llm_provider)
