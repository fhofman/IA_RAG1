import os
from langchain_chroma import Chroma
from config import PERSIST_DIR, COLLECTION_NAME


def persistir(chunks, embeddings, persist_dir=PERSIST_DIR):
    db_existed = os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0

    if db_existed:
        print("♻️ Índice existente detectado — cargando sin reindexar")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        print("🆕 No hay índice previo — indexando documentos por primera vez")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )

    print(f"📦 Documentos en la colección: {vectorstore._collection.count()}")
    return vectorstore
