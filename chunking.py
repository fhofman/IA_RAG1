from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generar_chunks(data_dir="data", chunk_size=500, chunk_overlap=50):
    # 1. Cargar todos los .txt de /data
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documentos_crudos = loader.load()

    print(f"📄 Documentos cargados: {len(documentos_crudos)}")

    # 2. Fragmentar respetando el mínimo pedido: 500 tokens, 50 de overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documentos_crudos)

    print(f"✂️ Fragmentos generados: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    chunks = generar_chunks()
    print("\n--- Ejemplo de fragmento ---")
    print(chunks[0].page_content[:300])
    print("Metadata:", chunks[0].metadata)
