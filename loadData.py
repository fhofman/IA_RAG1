import os

SOURCE_DIR = "source_data"
DATA_DIR = "data"


def cargar_datos(source_dir=SOURCE_DIR, data_dir=DATA_DIR):
    """Lee los .txt de source_dir como {nombre_archivo: contenido} y los escribe en data_dir."""
    os.makedirs(data_dir, exist_ok=True)

    documentos = {}
    for nombre_archivo in sorted(os.listdir(source_dir)):
        if not nombre_archivo.endswith(".txt"):
            continue
        with open(os.path.join(source_dir, nombre_archivo), "r", encoding="utf-8") as f:
            documentos[nombre_archivo] = f.read()

    for nombre_archivo, contenido in documentos.items():
        with open(os.path.join(data_dir, nombre_archivo), "w", encoding="utf-8") as f:
            f.write(contenido.strip())

    print(f"✅ {len(documentos)} archivos creados en /{data_dir}")
    return documentos


if __name__ == "__main__":
    cargar_datos()
