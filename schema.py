from pydantic import BaseModel, Field
from typing import List

class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar, parseado directamente de su output."""
    respuesta: str = Field(
        description="Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
                     "Si la información no está en el contexto, decir explícitamente que no se cuenta con esa información."
    )

class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response — combina el output del LLM con metadata verificable."""
    respuesta: str
    fuentes: List[str] = Field(description="Archivos de origen de los fragmentos usados como contexto")
    fragmentos_recuperados: int