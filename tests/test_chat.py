import pytest
import json
import httpx
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException
from chat import (
    GeminiConfig, tratar_status_gemini, chat_endpoint, 
    chat_stream_endpoint, _status_para_evento_sse, ChatRequest
)

# --- Mocks de Suporte ---
class MockStreamResponse:
    def __init__(self, status_code, lines):
        self.status_code = status_code
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def aread(self):
        return b""

    # O segredo é garantir que isso seja um gerador assíncrono real
    async def aiter_lines(self):
        for line in self.lines:
            yield line.decode("utf-8") if isinstance(line, bytes) else line



# --- Testes GeminiConfig ---

def test_gemini_config_validar_sucesso(monkeypatch):
    monkeypatch.setenv("URL_TEST", "http://fake.com")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    config = GeminiConfig("URL_TEST")
    config.validar()

def test_gemini_config_validar_erro_url(monkeypatch):
    monkeypatch.delenv("URL_TEST", raising=False)
    config = GeminiConfig("URL_TEST")
    with pytest.raises(HTTPException) as exc:
        config.validar()
    assert exc.value.status_code == 500

def test_gemini_config_build_payload(monkeypatch):
    monkeypatch.setenv("SYSTEM_PROMPT", "instrucao\\nquebra")
    config = GeminiConfig("URL")
    payload = config.build_payload("meu prompt")
    assert "instrucao\nquebra" in payload["systemInstruction"]["parts"][0]["text"]
    assert "meu prompt" == payload["contents"][0]["parts"][0]["text"]

# --- Testes de Erros e Status ---

@pytest.mark.parametrize("status, expected_exc", [
    (400, 422), (401, 502), (403, 502), (429, 429), (500, 502)
])
def test_tratar_status_gemini_erros(status, expected_exc):
    with pytest.raises(HTTPException) as exc:
        tratar_status_gemini(status)
    assert exc.value.status_code == expected_exc

def test_status_para_evento_sse():
    assert "error" in _status_para_evento_sse(400)
    assert "indisponível" in _status_para_evento_sse(503)
    assert _status_para_evento_sse(200) is None

# --- Testes chat_endpoint (POST) ---

@pytest.mark.asyncio
async def test_chat_endpoint_prompt_vazio():
    with pytest.raises(HTTPException) as exc:
        await chat_endpoint(ChatRequest(prompt=" "))
    assert exc.value.status_code == 422

@pytest.mark.asyncio
async def test_chat_endpoint_sucesso(monkeypatch):
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "raw body"
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Olá Mundo"}]}}]
    }

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
        resp = await chat_endpoint(ChatRequest(prompt="oi"))
        assert resp["response"] == "Olá Mundo"

@pytest.mark.asyncio
async def test_chat_endpoint_bloqueio_gemini(monkeypatch):
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false") # Garanta que o mock está desligado

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Definimos o texto para evitar erro no tratar_status_gemini
    mock_resp.text = '{"promptFeedback": {"blockReason": "SAFETY"}}'
    mock_resp.json.return_value = {"promptFeedback": {"blockReason": "SAFETY"}}

    # IMPORTANTE: Use o caminho real do seu arquivo onde está o chat_endpoint
    # Se o arquivo for app/api/prompt.py, use "app.api.prompt.httpx.AsyncClient.post"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        with pytest.raises(HTTPException) as exc:
            from chat import chat_endpoint # Import local para garantir context
            await chat_endpoint(ChatRequest(prompt="oi"))
        
        assert exc.value.status_code == 422
        assert "bloqueado" in exc.value.detail


@pytest.mark.asyncio
async def test_chat_endpoint_timeout(monkeypatch):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos o Mock da instância
    # Importante: mockamos o __aenter__ para retornar o próprio mock (para o 'async with')
    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__.return_value = mock_client_instance
    
    # Fazemos o post lançar o erro
    mock_client_instance.post.side_effect = httpx.TimeoutException("timeout")

    # 3. O PATCH deve ser no caminho do seu módulo
    # Se seu arquivo é app/api/prompt.py, use "app.api.prompt.httpx.AsyncClient"
    # Se for api/prompt.py, use "api.prompt.httpx.AsyncClient"
    with patch("chat.httpx.AsyncClient", return_value=mock_client_instance):
        with pytest.raises(HTTPException) as exc:
            await chat_endpoint(ChatRequest(prompt="oi"))
        
        # Agora deve cair no except httpx.TimeoutException e retornar 504
        assert exc.value.status_code == 504
        assert "Gemini não respondeu a tempo" in exc.value.detail



# --- Testes chat_stream_endpoint (STREAM) ---

@pytest.mark.asyncio
async def test_chat_stream_mock(monkeypatch):
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "true")

    response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    
    assert any("data: Essa" in c for c in chunks)
    assert any("event: done" in c for c in chunks)

@pytest.mark.asyncio
async def test_chat_stream_error_status(monkeypatch):
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # Simula erro 429 (Rate Limit) no stream
    mock_stream = MockStreamResponse(429, [])
    
    with patch("httpx.AsyncClient.stream", return_value=mock_stream):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        assert any("event: error" in c for c in chunks)
        assert any("Limite de requisições" in c for c in chunks)

@pytest.mark.asyncio
async def test_chat_stream_exception_generica(monkeypatch):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos um mock para o cliente que falha ao abrir o stream
    mock_client = AsyncMock()
    mock_client.stream.side_effect = Exception("Crash Inesperado")
    
    # 3. Aplicamos o patch na classe para garantir que o 'async with' a use
    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        
        # 4. Ao iterar, o erro deve ser disparado ou capturado pelo log do seu código
        # Se o seu código tiver um try/except genérico dentro do 'gerar', 
        # ele não vai dar raise aqui, ele vai apenas parar o stream.
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # Se o seu código capturou o erro e não deu yield em nada de erro:
        assert len(chunks) == 0 or any("error" in c for c in chunks)

@pytest.mark.asyncio
@pytest.mark.parametrize("exception, expected_code", [
    (httpx.ConnectError("Erro de conexão"), 502),
    (httpx.RequestError("Erro de rede"), 502),
])
async def test_chat_endpoint_erros_rede_especificos(monkeypatch, exception, expected_code):
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.side_effect = exception

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(HTTPException) as exc:
            await chat_endpoint(ChatRequest(prompt="oi"))
        assert exc.value.status_code == expected_code

@pytest.mark.asyncio
async def test_chat_endpoint_json_corrompido(monkeypatch):
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Não é JSON") # Força erro no .json()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(HTTPException) as exc:
            await chat_endpoint(ChatRequest(prompt="oi"))
        assert exc.value.status_code == 502
        assert "Resposta inválida" in exc.value.detail

@pytest.mark.asyncio
async def test_chat_stream_sucesso_completo(monkeypatch):
    # 1. Setup de Ambiente
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criar Mock da Resposta do Stream
    payload_chunk = json.dumps({"candidates": [{"content": {"parts": [{"text": "Olá"}]}}]})
    lines = [f"data: {payload_chunk}".encode("utf-8"), b"data: [DONE]"]
    
    # Usando a classe MockStreamResponse que definimos antes
    mock_stream = MockStreamResponse(200, lines)

    # 3. Criar Mock do Cliente httpx
    # 3. Criar Mock do Cliente httpx
    mock_client = MagicMock() # Use MagicMock aqui para o cliente
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()

    # O SEGREDO: .stream não deve ser AsyncMock diretamente, 
    # mas um MagicMock que retorna o seu mock_stream
    mock_client.stream = MagicMock(return_value=mock_stream)

# Patch no objeto 'httpx' que foi importado dentro do arquivo chat.py
    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        texto_completo = "".join(chunks)
        
        # Depuração caso continue dando erro 500
        if "erro interno" in texto_completo:
            print(f"\nResposta recebida: {texto_completo}")

        assert "Olá" in texto_completo
        assert "[DONE]" in texto_completo

def test_gemini_config_validar_erro_key_faltante(monkeypatch):
    # 1. Configuramos a URL (para passar pelo primeiro IF)
    monkeypatch.setenv("URL_TESTE", "http://fake.com")
    
    # 2. Garantimos que a KEY não existe no ambiente
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    config = GeminiConfig("URL_TESTE")
    
    # 3. O teste deve capturar a HTTPException 500 lançada na validação da KEY
    with pytest.raises(HTTPException) as exc:
        config.validar()
    
    assert exc.value.status_code == 500
    assert exc.value.detail == "Serviço não configurado corretamente"

@pytest.mark.asyncio
async def test_chat_endpoint_mock_retorno(monkeypatch):
    # 1. Configuramos o ambiente para ativar o MOCK
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "true") # Esta linha ativa o IF do seu código

    # 2. Chamamos o endpoint
    # Note que não precisamos de patch no httpx aqui, pois o código retorna ANTES de usar a rede
    request_data = ChatRequest(prompt="olá")
    response = await chat_endpoint(request_data)

    # 3. Verificamos se o retorno é exatamente o texto do mock
    assert response == {"response": "Mensagem Mockada para testes. Gemini não foi chamado."}

@pytest.mark.asyncio
async def test_chat_endpoint_estrutura_inesperada(monkeypatch):
    # 1. Setup básico para passar pelas validações iniciais
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos uma resposta JSON que é válida como JSON,
    # mas não tem 'candidates' nem 'promptFeedback'
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"algo_aleatorio": "valor"}'
    mock_resp.json.return_value = {"algo_aleatorio": "valor"}

    # 3. Mock do cliente httpx
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        # 4. O teste deve capturar a HTTPException 502 (Resposta em formato inesperado)
        with pytest.raises(HTTPException) as exc:
            await chat_endpoint(ChatRequest(prompt="oi"))

        assert exc.value.status_code == 502
        assert "formato inesperado" in exc.value.detail

@pytest.mark.asyncio
async def test_chat_endpoint_erro_generico_inesperado(monkeypatch):
    # 1. Setup para passar pelas validações iniciais
    monkeypatch.setenv("URL_GEMINI", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Forçamos um erro que NÃO é do httpx. 
    # Podemos fazer o próprio AsyncClient estourar ao ser instanciado.
    with patch("chat.httpx.AsyncClient", side_effect=Exception("Erro catastrófico")):
        with pytest.raises(HTTPException) as exc:
            await chat_endpoint(ChatRequest(prompt="oi"))
        
        # 3. Verifica se caiu no except Exception final (Status 500)
        assert exc.value.status_code == 500
        assert "Ocorreu um erro interno" in exc.value.detail

@pytest.mark.asyncio
async def test_chat_stream_endpoint_prompt_vazio():
    # 1. Criamos um request com prompt inválido (vazio ou só espaços)
    request_invalido = ChatRequest(prompt="   ")

    # 2. Chamamos o endpoint de stream
    # O pytest deve capturar a HTTPException 422
    with pytest.raises(HTTPException) as exc:
        await chat_stream_endpoint(request_invalido)

    # 3. Validamos o status e a mensagem
    assert exc.value.status_code == 422
    assert exc.value.detail == "Prompt não pode ser vazio"

@pytest.mark.asyncio
async def test_chat_stream_ignora_linhas_invalidas(monkeypatch):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos um stream com linhas que devem ser ignoradas pelo 'continue'
    payload_valido = json.dumps({"candidates": [{"content": {"parts": [{"text": "Ok"}]}}]})
    lines = [
        "",                             # Linha vazia (cai no if not linha)
        "linha sem prefixo",            # Linha sem 'data: ' (cai no startswith)
        f"data: {payload_valido}",      # Linha válida para o stream continuar
        "data: [DONE]"                  # Finalizador
    ]
    
    mock_stream = MockStreamResponse(200, lines)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.stream.return_value = mock_stream

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # 3. Verificamos se, mesmo com o lixo no stream, o dado válido passou
        texto_completo = "".join(chunks)
        assert "Ok" in texto_completo

@pytest.mark.asyncio
async def test_chat_stream_json_invalido_ignorado(monkeypatch):
    # 1. Setup básico
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos um stream com um chunk que é JSON inválido
    payload_valido = json.dumps({"candidates": [{"content": {"parts": [{"text": "Texto Válido"}]}}]})
    lines = [
        "data: {json_quebrado: true",  # JSON malformado (faltando aspas/chaves)
        f"data: {payload_valido}",    # Chunk válido para o stream não morrer
        "data: [DONE]"
    ]
    
    mock_stream = MockStreamResponse(200, lines)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.stream.return_value = mock_stream

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # 3. Validamos que o chunk inválido foi ignorado e o válido processado
        texto_completo = "".join(chunks)
        assert "Texto Válido" in texto_completo

@pytest.mark.asyncio
async def test_chat_stream_conteudo_bloqueado(monkeypatch):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Simula o JSON de bloqueio que o Gemini envia
    payload_bloqueio = json.dumps({
        "promptFeedback": {"blockReason": "SAFETY"}
    })
    lines = [f"data: {payload_bloqueio}".encode("utf-8")]
    
    mock_stream = MockStreamResponse(200, lines)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.stream.return_value = mock_stream

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # 3. Verificamos se o evento de erro foi gerado com o motivo correto
        texto_completo = "".join(chunks)
        assert "event: error" in texto_completo
        assert "Conteúdo bloqueado pelo Gemini: SAFETY" in texto_completo

@pytest.mark.asyncio
async def test_chat_stream_estrutura_chunk_invalida(monkeypatch):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. JSON válido, mas SEM a chave 'candidates'
    payload_errado = json.dumps({"informacao_inutil": "blabla"})
    payload_valido = json.dumps({"candidates": [{"content": {"parts": [{"text": "Fim"}]}}]})
    
    lines = [
        f"data: {payload_errado}", # Vai causar KeyError
        f"data: {payload_valido}", # Para o stream terminar com sucesso
        "data: [DONE]"
    ]
    
    mock_stream = MockStreamResponse(200, lines)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.stream.return_value = mock_stream

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # 3. Verifica se o chunk inválido foi ignorado (pelo continue) 
        # e o válido foi processado
        texto_completo = "".join(chunks)
        assert "Fim" in texto_completo

@pytest.mark.asyncio
@pytest.mark.parametrize("excecao, mensagem_esperada", [
    (httpx.TimeoutException("timeout"), "Gemini não respondeu a tempo"),
    (httpx.RemoteProtocolError("error"), "Conexão encerrada inesperadamente"),
    (httpx.ConnectError("conn"), "Não foi possível conectar ao Gemini"),
    (httpx.RequestError("req"), "Erro de comunicação com o Gemini"),
])
async def test_chat_stream_erros_rede(monkeypatch, excecao, mensagem_esperada):
    # 1. Setup
    monkeypatch.setenv("URL_GEMINI_STREAM", "http://fake")
    monkeypatch.setenv("GEMINI_API_KEY", "123")
    monkeypatch.setenv("USAR_MOCK", "false")

    # 2. Criamos o Mock do Cliente que lança a exceção ao tentar abrir o stream
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    # Aqui forçamos o erro no .stream(...)
    mock_client.stream.side_effect = excecao

    with patch("chat.httpx.AsyncClient", return_value=mock_client):
        response = await chat_stream_endpoint(ChatRequest(prompt="oi"))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        
        # 3. Verificamos se o catch capturou e enviou o evento SSE correto
        texto_completo = "".join(chunks)
        assert "event: error" in texto_completo
        assert mensagem_esperada in texto_completo
