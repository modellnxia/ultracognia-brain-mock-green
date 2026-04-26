import asyncio
import json
import logging
import traceback
import os
from urllib import response

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prompt", tags=["prompt"])

MOCK_TEXTO = (
    "Essa é uma resposta mockada para testes. O Gemini não foi chamado. "
    "Com base na análise dos dados fornecidos, identifiquei um padrão significativo "
    "nas interações recentes. O processamento de linguagem natural indica que a intenção "
    "do usuário está voltada para a otimização de fluxos de trabalho internos."
)

class ChatRequest(BaseModel):
    prompt: str
    agente: int | None

class GeminiConfig:
    def __init__(self, url_env: str):
        self.url         = os.getenv(url_env)
        self.key         = os.getenv("GEMINI_API_KEY")
        self.usar_mock   = os.getenv("USAR_MOCK", "false").lower() == "true"
        self._url_env    = url_env

    def validar(self):
        if not self.url:
            logger.error("Variável %s não configurada", self._url_env)
            raise HTTPException(status_code=500, detail="Serviço não configurado corretamente")
        if not self.key:
            logger.error("Variável GEMINI_API_KEY não configurada")
            raise HTTPException(status_code=500, detail="Serviço não configurado corretamente")

    def build_payload(self, prompt: str, agente: int | None) -> dict:
        if agente == 1:
            self.system_prompt = os.getenv("SYSTEM_PROMPT_1", "").replace("\\n", "\n")
        elif agente == 2:
            self.system_prompt = os.getenv("SYSTEM_PROMPT_2", "").replace("\\n", "\n")
        else:
            self.system_prompt = os.getenv("SYSTEM_PROMPT", "").replace("\\n", "\n")
        return {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents":          [{"parts": [{"text": prompt}]}],
        }

def tratar_status_gemini(status_code: int, body_preview: str = "") -> None:
    """Lança HTTPException para status de erro do Gemini. Não faz nada para 2xx."""
    if status_code == 400:
        logger.warning("Requisição inválida para o Gemini: %s", body_preview)
        raise HTTPException(status_code=422, detail="Prompt inválido ou mal formatado")

    if status_code in (401, 403):
        logger.warning("API Key do Gemini inválida ou sem permissão: %s", status_code)
        raise HTTPException(status_code=502, detail="Erro de autenticação com o Gemini")

    if status_code == 429:
        logger.warning("Limite de requisições do Gemini atingido")
        raise HTTPException(status_code=429, detail="Limite de requisições atingido. Tente novamente em instantes.")

    if status_code >= 500:
        logger.error("Gemini retornou erro interno: %s", status_code)
        raise HTTPException(status_code=502, detail="Serviço do Gemini indisponível")

async def _call_gemini(client: httpx.AsyncClient, config: GeminiConfig, prompt: str, agente: int | None) -> httpx.Response:
    try:
        return await client.post(
            f"{config.url}?key={config.key}",
            headers={"Content-Type": "application/json"},
            json=config.build_payload(prompt, agente),
            timeout=60.0,
        )
    except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
        raise


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt não pode ser vazio")

    config = GeminiConfig("URL_GEMINI")
    config.validar()

    if config.usar_mock:
        return {"response": "Mensagem Mockada para testes. Gemini não foi chamado."}

    try:
        async with httpx.AsyncClient() as client:
            response = await _call_gemini(client, config, request.prompt, request.agente)

        tratar_status_gemini(response.status_code, response.text[:200])

        try:
            data = response.json()
        except Exception:
            logger.error("Gemini retornou resposta inválida: %s", response.text[:200])
            raise HTTPException(status_code=502, detail="Resposta inválida do Gemini")

        try:
            texto = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            if "promptFeedback" in data:
                motivo = data["promptFeedback"].get("blockReason", "desconhecido")
                logger.warning("Prompt bloqueado pelo Gemini: %s", motivo)
                raise HTTPException(status_code=422, detail=f"Prompt bloqueado pelo Gemini: {motivo}")
            logger.error("Estrutura inesperada na resposta do Gemini: %s", data)
            raise HTTPException(status_code=502, detail="Resposta do Gemini em formato inesperado")

        return {"response": texto}

    except httpx.TimeoutException:
        logger.error("Timeout ao chamar o Gemini após 60s")
        raise HTTPException(status_code=504, detail="Gemini não respondeu a tempo. Tente novamente.")

    except httpx.ConnectError:
        logger.error("Não foi possível conectar ao Gemini em %s", config.url)
        raise HTTPException(status_code=502, detail="Não foi possível conectar ao Gemini")

    except httpx.RequestError as e:
        logger.error("Erro de rede ao chamar o Gemini: %s", e)
        raise HTTPException(status_code=502, detail="Erro de comunicação com o Gemini")

    except HTTPException:
        raise

    except Exception:
        logger.error("Erro inesperado: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno. Tente novamente.")

def _status_para_evento_sse(status_code: int) -> str | None:
    """Retorna string de evento SSE de erro, ou None se status for OK."""
    mapa = {
        400: "Prompt inválido ou mal formatado",
        401: "Erro de autenticação com o Gemini",
        403: "Erro de autenticação com o Gemini",
        429: "Limite de requisições atingido. Tente novamente.",
    }
    if status_code in mapa:
        return f"event: error\ndata: {mapa[status_code]}\n\n"
    if status_code >= 500:
        return "event: error\ndata: Serviço do Gemini indisponível\n\n"
    return None


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt não pode ser vazio")

    config = GeminiConfig("URL_GEMINI_STREAM")
    config.validar()

    async def gerar():
        if config.usar_mock:
            for palavra in MOCK_TEXTO.split():
                yield f"data: {palavra} \n\n"
                await asyncio.sleep(0.15)
            yield "event: done\ndata: [DONE]\n\n"
            return

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{config.url}&key={config.key}",
                    headers={"Content-Type": "application/json"},
                    json=config.build_payload(request.prompt, request.agente),
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        corpo_erro = await response.aread() # Lê o corpo do erro do Gemini
                        logger.error(f"Erro Gemini API ({response.status_code}): {corpo_erro.decode()}")
                    
                    evento_erro = _status_para_evento_sse(response.status_code)
                    if evento_erro:
                        yield evento_erro
                        return

                    async for linha in response.aiter_lines():
                        if not linha or not linha.startswith("data: "):
                            continue

                        chunk = linha.removeprefix("data: ")

                        if chunk == "[DONE]":
                            yield "event: done\ndata: [DONE]\n\n"
                            break

                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            logger.warning("Chunk inválido ignorado: %s", chunk[:100])
                            continue

                        if "promptFeedback" in data:
                            motivo = data["promptFeedback"].get("blockReason", "desconhecido")
                            logger.warning("Prompt bloqueado pelo Gemini: %s", motivo)
                            yield f"event: error\ndata: Conteúdo bloqueado pelo Gemini: {motivo}\n\n"
                            return

                        try:
                            token = data["candidates"][0]["content"]["parts"][0]["text"]
                            if token:
                                yield f"data: {token}\n\n"
                        except (KeyError, IndexError):
                            logger.warning("Estrutura inesperada no chunk: %s", data)
                            continue

        except httpx.TimeoutException:
            logger.error("Timeout no stream do Gemini")
            yield "event: error\ndata: Gemini não respondeu a tempo\n\n"

        except httpx.RemoteProtocolError:
            logger.error("Conexão com o Gemini encerrada inesperadamente")
            yield "event: error\ndata: Conexão encerrada inesperadamente\n\n"

        except httpx.ConnectError:
            logger.error("Não foi possível conectar ao Gemini")
            yield "event: error\ndata: Não foi possível conectar ao Gemini\n\n"

        except httpx.RequestError as e:
            logger.error("Erro de rede no stream: %s", e)
            yield "event: error\ndata: Erro de comunicação com o Gemini\n\n"

        except Exception:
            logger.error("Erro inesperado no stream: %s", traceback.format_exc())
            yield "event: error\ndata: Ocorreu um erro interno. Tente novamente.\n\n"

    return StreamingResponse(
        gerar(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )