import os
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
from urllib3 import request
from chat import router as chat_router
from fastapi.responses import JSONResponse
import logging

load_dotenv()

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000) #compacta acima de 1kb

# Registra as rotas de chat
app.include_router(chat_router)

@app.get("/")
def home():
    return {"status": "API Online"}

@app.middleware("http")
async def validar_acesso(request: Request, call_next):
    api_key = request.headers.get("x-api-key");
    API_KEY = os.getenv("API_KEY", "")

    if api_key != API_KEY:
        print(f"Headers recebidos: {dict(request.headers)}")
        print(f"Chaves: {api_key} - {API_KEY}")
        return JSONResponse(status_code=403, content={"detail": "Não autorizado"})

    return await call_next(request)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]  # garante saída no stdout
)