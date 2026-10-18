import os
import sys
from getpass import getpass

from dotenv import load_dotenv

# La clave se lee del archivo .env (que NO se sube al repo; ver .env.example)
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    if not sys.stdin.isatty():
        raise RuntimeError("Falta GOOGLE_API_KEY: definila en el archivo .env (ver .env.example)")
    os.environ["GOOGLE_API_KEY"] = getpass("🔑 Ingresá tu GOOGLE_API_KEY (gratis en aistudio.google.com/apikey): ").strip()
