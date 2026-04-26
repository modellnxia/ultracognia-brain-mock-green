import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app  # Certifique-se que o nome do arquivo seja main.py

client = TestClient(app)

# 1. Teste da Rota Home (Sucesso)
def test_home_success(monkeypatch):
    # Configuramos uma chave para passar pelo middleware
    monkeypatch.setenv("API_KEY", "secret123")
    
    response = client.get("/", headers={"x-api-key": "secret123"})
    
    assert response.status_code == 200
    assert response.json() == {"status": "API Online"}

# 2. Teste de Acesso Negado (Middleware)
def test_validar_acesso_forbidden(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    
    # Enviamos uma chave errada
    response = client.get("/", headers={"x-api-key": "wrong_key"})
    
    assert response.status_code == 403
    assert response.json() == {"detail": "Não autorizado"}

# 3. Teste de Acesso sem Header (Middleware)
def test_validar_acesso_no_header(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    
    # Não enviamos o header x-api-key
    response = client.get("/")
    
    assert response.status_code == 403

# 4. Teste de API_KEY Vazia no ENV (Caso de borda)
def test_validar_acesso_empty_env(monkeypatch):
    # Se o ENV estiver vazio e o header vier vazio, deve passar (conforme seu código)
    monkeypatch.setenv("API_KEY", "")
    
    response = client.get("/", headers={"x-api-key": ""})
    
    assert response.status_code == 200

# 5. Teste do Middleware GZip
def test_gzip_compression(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    
    # Criamos uma resposta longa para acionar o GZip (mínimo 1000 bytes)
    # Vamos testar através de uma rota que aceite o header x-api-key
    # Como a rota "/" retorna pouco, o GZip não agiria. 
    # Mas podemos validar se o middleware está presente na stack do app.
    
    from fastapi.middleware.gzip import GZipMiddleware
    has_gzip = any(isinstance(m.cls, type) and m.cls == GZipMiddleware for m in app.user_middleware)
    assert has_gzip is True

# 6. Teste de Integração com o Router de Chat (Verifica se está incluído)
def test_chat_router_included(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    
    # Tenta acessar uma rota que sabemos que existe no chat_router
    # apenas para ver se o prefixo /prompt foi registrado
    response = client.post("/prompt/chat", 
                           headers={"x-api-key": "secret123"},
                           json={"prompt": ""}) # prompt vazio gera 422, mas prova que a rota existe
    
    assert response.status_code == 422
