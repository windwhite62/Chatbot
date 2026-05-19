"""Tests unitaires — Chatbot Lambersart backend"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock de l'API Mistral pour les tests
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("ADMIN_TOKEN", "ci-token")

from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"

def test_chat_empty_message(client):
    r = client.post("/chat", json={"session_id": "test", "message": ""})
    assert r.status_code == 400

def test_off_topic_blocked(client):
    r = client.post("/chat", json={"session_id": "test", "message": "bitcoin crypto"})
    assert r.status_code == 200
    data = r.get_json()
    assert "Lambersart" in data["answer"]

def test_reindex_unauthorized(client):
    r = client.post("/reindex")
    assert r.status_code == 401

def test_reindex_authorized(client):
    r = client.post("/reindex", headers={"X-Admin-Token": "ci-token"})
    # Peut échouer si pas de réseau, mais ne doit pas lever 500 d'auth
    assert r.status_code in (200, 500)
