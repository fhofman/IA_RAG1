"""Pruebas del punto 4 de la consigna.

Usan los proveedores de config.py / .env (EMBEDDING_PROVIDER y LLM_PROVIDER).
Ejecutar con:  pytest -v test_rag.py
"""
import asyncio

from response import get_rag_response

RESPUESTA_SIN_CONTEXTO = "No tengo acceso a esa información"


def test_pregunta_con_respuesta_en_los_documentos():
    r = asyncio.run(get_rag_response("¿Cuánto dura la batería de los Auralis X2 con ANC activado?"))
    print(r.model_dump_json(indent=2))

    assert "48" in r.respuesta
    assert "data/auralis_x2_auriculares.txt" in r.fuentes
    assert r.fragmentos_recuperados > 0


def test_pregunta_trampa_no_alucina():
    # El Krono R11 no existe: los documentos solo hablan del Krono R5, R7 y R9
    r = asyncio.run(get_rag_response("¿Cuál es la potencia de succión del Krono R11?"))
    print(r.model_dump_json(indent=2))

    assert RESPUESTA_SIN_CONTEXTO in r.respuesta
