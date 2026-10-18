# IA_RAG1 — RAG sobre fichas de productos

Pipeline RAG (Retrieval-Augmented Generation) con LangChain, ChromaDB y Pydantic.
Responde preguntas sobre tres productos tecnológicos **ficticios** (así se puede
comprobar que las respuestas salen de los documentos y no del conocimiento previo del LLM):

- **Auralis X2**: auriculares con cancelación de ruido
- **Krono R9**: robot aspiradora y fregona
- **Vértice Book 14 Ultra**: notebook ultraliviana

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración (`.env`)

Las claves **no** están en el código: se leen de un archivo `.env`, que está en el
`.gitignore` y nunca se sube al repo.

```bash
cp .env.example .env
# editá .env y completá GOOGLE_API_KEY (gratis en https://aistudio.google.com/apikey)
```

En el `.env` también se pueden elegir los proveedores por defecto (`EMBEDDING_PROVIDER`, `LLM_PROVIDER`).

## Uso

```bash
python main.py                       # proveedores por defecto (config.py / .env) y todas las preguntas de prueba
python main.py gemini                # embeddings y LLM de Gemini
python main.py huggingface "¿Qué significa el Error 6 del Krono R9?"   # una sola pregunta
python main.py docker "" docker      # 100% local con Docker Model Runner
```

Argumentos: `python main.py [proveedor_embeddings] ["pregunta"] [proveedor_llm]`.

Desde código:

```python
import asyncio
from response import get_rag_response

r = asyncio.run(get_rag_response("¿Cuánto dura la batería de los Auralis X2?"))
print(r.respuesta, r.fuentes)
```

## Tests (pregunta con respuesta y pregunta trampa)

```bash
pytest -v test_rag.py
```

- `test_pregunta_con_respuesta_en_los_documentos`: la respuesta tiene que incluir "48" (horas) y citar el documento de los Auralis X2.
- `test_pregunta_trampa_no_alucina`: pregunta por un producto que no existe (Krono R11); el modelo tiene que responder "No tengo acceso a esa información…".

## Cómo funciona

| Paso | Archivo | Qué hace |
|---|---|---|
| 1. Carga | `loadData.py` | Copia los `.txt` de `source_data/` a `data/` |
| 2. Chunking | `chunking.py` | `DirectoryLoader` + `RecursiveCharacterTextSplitter` por tokens: fragmentos de hasta 500 tokens con 50 de overlap |
| 3. Embeddings | `embeddings.py` | Modelo de embeddings según el proveedor |
| 4. Persistencia | `persistData.py` | Índice ChromaDB local en `vectorstore/<proveedor>/`; si ya existe, lo reutiliza |
| 5. Respuesta | `response.py` | `get_rag_response(query)` asíncrona: búsqueda de similitud → prompt con los fragmentos → LLM asíncrono → modelo Pydantic |
| Orquestador | `main.py` | Ejecuta los pasos en orden |

La respuesta es un `RAGResponse` (`schema.py`) con el texto (`respuesta`), las
referencias (`fuentes`, que salen de la metadata de los fragmentos recuperados y no
del LLM, así no se pueden alucinar) y la cantidad de fragmentos usados.

El LLM devuelve su salida con `with_structured_output(..., method="json_schema")`:
el servidor restringe la generación al esquema Pydantic, lo que es más robusto que
pedir el formato en el prompt (sobre todo con modelos locales chicos).

## Proveedores

| Proveedor | Embeddings | LLM | Requiere |
|---|---|---|---|
| `gemini` | `gemini-embedding-001` | `gemini-flash-latest` | `GOOGLE_API_KEY` |
| `huggingface` | `paraphrase-multilingual-MiniLM-L12-v2` (local, CPU) | — | nada (descarga ~470 MB la primera vez) |
| `docker` | `ai/embeddinggemma` | `ai/llama3.2:3B-Q4_K_M` | Docker Model Runner en el puerto 12434 |

Para el proveedor `docker`:

```bash
docker desktop enable model-runner --tcp 12434
docker model pull ai/embeddinggemma
docker model pull ai/llama3.2:3B-Q4_K_M
```

⚠️ Se usa el mismo modelo de embeddings para indexar y para consultar. Por eso cada
proveedor tiene su propio índice. Si cambiás los documentos de `source_data/`, borrá
`vectorstore/<proveedor>/` para reindexar.
